"""Dry run test for encoder-prefill-decode pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), 'astra-sim'))

from serving.core.request import Request, Batch
from serving.core.encoder_model import EncoderLatencyModel
from serving.core.trace_generator import generate_encoder_trace
from serving.core.config_builder import build_cluster_config
from serving.core.router import Router
from serving.core.utils import get_config


def test_config_builder():
    print("=" * 60)
    print("TEST 1: Config Builder with encoder pd_type")
    print("=" * 60)
    cluster = build_cluster_config('.', 'configs/cluster/single_node_encoder_pd_instance.json')
    print(f"  Num instances: {cluster['num_instances']}")
    for inst in cluster['instances']:
        print(f"  Instance {inst['instance_id']}: pd_type={inst['pd_type']}, "
              f"tp={inst['tp_size']}, npus={inst['num_npus']}")
    print(f"  Inst2NPU mapping: {cluster['inst2npu_mapping']}")
    print(f"  Total NPUs: {cluster['total_npu']}")
    print("  PASSED\n")


def test_router():
    print("=" * 60)
    print("TEST 2: Router multimodal routing")
    print("=" * 60)

    class MockScheduler:
        def __init__(self, pd_type, instance_id):
            self.pd_type = pd_type
            self.instance_id = instance_id
            self.model = 'meta-llama/Llama-3.1-8B'
            self.request = []
            self.inflight = []
        def add_request(self, req, is_init=True, image_tokens=0, num_images=0, image_resolution=0):
            self.request.append({'image_tokens': image_tokens, 'num_images': num_images})
        def add_request_from_encoder(self, req):
            self.request.append({'from_encoder': True})
        def add_decode(self, req):
            self.request.append({'decode': True})

    encoder_sched = MockScheduler('encoder', 0)
    prefill_sched = MockScheduler('prefill', 1)
    decode_sched = MockScheduler('decode', 2)
    schedulers = [encoder_sched, prefill_sched, decode_sched]

    router = Router(num_instances=3, schedulers=schedulers, req_num=20, routing_policy='RR')
    router.load_requests('workloads/multimodal-test-20.jsonl')
    routed = router.route_arrived_requests(float('inf'))

    print(f"  Routed {routed} requests total")
    print(f"  Encoder queue: {len(encoder_sched.request)} (multimodal)")
    print(f"  Prefill queue: {len(prefill_sched.request)} (text-only)")
    print(f"  Decode queue:  {len(decode_sched.request)}")
    assert len(encoder_sched.request) > 0, "No multimodal requests routed to encoder!"
    assert all(r['image_tokens'] > 0 for r in encoder_sched.request), "Encoder got text-only request!"
    assert all(r['image_tokens'] == 0 for r in prefill_sched.request), "Prefill got multimodal request!"
    print("  PASSED\n")


def test_encoder_trace():
    print("=" * 60)
    print("TEST 3: Encoder trace generation (analytical model)")
    print("=" * 60)

    config = get_config('meta-llama/Llama-3.1-8B')

    batch = Batch(batch_id=0, model='meta-llama/Llama-3.1-8B',
                  total_len=1152, kv_len=0, q_list=[576, 576], k_list=[],
                  num_prefill=2, num_decode=0, prefill_q_list=[576, 576],
                  prefill_k_list=[], decode_k_list=[], batch_time=0, kv_size=0)

    req1 = Request(0, 'meta-llama/Llama-3.1-8B', 666, 778, 480841720, 0)
    req1.image_tokens = 576
    req1.num_images = 1

    req2 = Request(1, 'meta-llama/Llama-3.1-8B', 1242, 1481, 948079651, 0)
    req2.image_tokens = 1152
    req2.num_images = 2

    batch.requests = [req1, req2]

    output_path = generate_encoder_trace(batch, 'RTXPRO6000', node_id=0,
                                         instance_id=0, config=config, fp=2)

    with open(output_path) as f:
        lines = f.readlines()

    # Parse trace
    assert lines[0].startswith("ENCODER"), f"Wrong header: {lines[0]}"
    num_layers = int(lines[1].strip())
    assert num_layers == 99, f"Expected 99 layers, got {num_layers}"

    # Check first layer input is from REMOTE (image data from CPU)
    first_layer = lines[3].split()
    assert 'REMOTE:0' in first_layer, f"First layer should have REMOTE input: {first_layer}"

    # Check last layer output is to REMOTE (embeddings to prefill)
    last_layer = lines[-1].split()
    assert 'REMOTE:0' in last_layer, f"Last layer should have REMOTE output: {last_layer}"

    # Total compute latency
    total_ns = 0
    for line in lines[3:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                total_ns += int(parts[1])
            except ValueError:
                pass

    print(f"  Trace file: {output_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Total images: 3 (1 + 2)")
    print(f"  Total compute: {total_ns} ns = {total_ns/1e6:.2f} ms")
    print(f"  First layer: {' '.join(first_layer[:4])}")
    print(f"  Last layer: {' '.join(last_layer[:4])}")
    print("  PASSED\n")


def test_encoder_latency_model():
    print("=" * 60)
    print("TEST 4: Encoder latency model scaling")
    print("=" * 60)

    encoder = EncoderLatencyModel('RTXPRO6000')
    print(f"  ViT-L/14 @ 336px on RTXPRO6000:")
    print(f"  Patches per image: {encoder.num_patches}")
    print(f"  Encoder weight size: {encoder.get_encoder_weight_bytes()/1e6:.1f} MB")
    print()
    print(f"  Latency scaling:")
    for n in [1, 2, 4, 8, 16]:
        lat_ms = encoder.estimate_total_latency_us(n) / 1000
        per_img = lat_ms / n
        print(f"    {n:2d} image(s): {lat_ms:7.2f} ms total, {per_img:.2f} ms/image")
    print()

    # Test H100
    encoder_h100 = EncoderLatencyModel('H100')
    lat1 = encoder_h100.estimate_total_latency_us(1) / 1000
    print(f"  H100: 1 image = {lat1:.2f} ms (vs RTXPRO6000: {encoder.estimate_total_latency_us(1)/1000:.2f} ms)")
    assert lat1 < encoder.estimate_total_latency_us(1) / 1000, "H100 should be faster!"
    print("  PASSED\n")


def test_transfer_flow():
    print("=" * 60)
    print("TEST 5: Encoder -> Prefill transfer")
    print("=" * 60)

    # Simulate: encoder completes, request transfers to prefill
    req = Request(42, 'meta-llama/Llama-3.1-8B', 666, 778, 0, 0)
    req.image_tokens = 576
    req.num_images = 1
    req.encoder_done = False

    # After encoder is done:
    req.encoder_done = True

    assert req.encoder_done == True
    assert req.image_tokens == 576
    # Request should still be in prefill phase (num_computed_tokens == 0)
    assert req.is_prefill() == True
    print(f"  Request #{req.id}: encoder_done={req.encoder_done}, "
          f"image_tokens={req.image_tokens}, is_prefill={req.is_prefill()}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_config_builder()
    test_router()
    test_encoder_trace()
    test_encoder_latency_model()
    test_transfer_flow()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
