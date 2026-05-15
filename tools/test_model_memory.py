"""Test memory impact of embedding model loading."""
import os, time, psutil

def snap(label):
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1024**2
    vm = psutil.virtual_memory()
    print(f"[{label:35s}] rss={rss:.0f}MB sys={vm.used/1024**3:.1f}/{vm.total/1024**3:.1f}GB")
    return rss

snap("baseline")

from raggy_mcp.settings import EmbeddingProviderSettings
from raggy_mcp.embeddings.factory import create_embedding_provider

# Test 0.6B model (smallest)
print("\n--- Qwen3-0.6B ---")
es_small = EmbeddingProviderSettings(model_name="Qwen/Qwen3-Embedding-0.6B")
pp = create_embedding_provider(es_small)
snap("after create_0.6B")
if hasattr(pp, 'warm_up'):
    try: pp.warm_up()
    except: pass
snap("after warmup_0.6B")

# Test 4B model (current default)
print("\n--- Qwen3-4B ---")
es_4b = EmbeddingProviderSettings(model_name="Qwen/Qwen3-Embedding-4B")
pp4 = create_embedding_provider(es_4b)
snap("after create_4B")
if hasattr(pp4, 'warm_up'):
    try: pp4.warm_up()
    except: pass
snap("after warmup_4B")

# Test 8B model  
print("\n--- Qwen3-8B ---")
es_8b = EmbeddingProviderSettings(model_name="Qwen/Qwen3-Embedding-8B")
pp8 = create_embedding_provider(es_8b)
snap("after create_8B")
if hasattr(pp8, 'warm_up'):
    try: pp8.warm_up()
    except: pass
snap("after warmup_8B")

# Try an actual embedding with 8B
print("\n--- Embedding with 8B ---")
import asyncio
async def test():
    vec = await pp8.embed_query("what are some things adler said")
    snap("after embed_8B")
    print(f"  Vector dims: {len(vec)}")
asyncio.run(test())

snap("final")
print("\nDone.")
