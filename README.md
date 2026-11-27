# Lila ASI v2.6 (`lila-asi`) — Active System Image + Universal `.asiv` Vision Runtime

`lila-asi` is a standalone C11 + Python inference runtime and cartridge format (`.asi` for LLMs, `.asiv` for vision encoders/projectors) built for fast local CPU inference on **hybrid Gated DeltaNet + Full-Attention** models (`Qwen3.5`, `Qwen3.5-VL`), **dense Transformers** (`Llama`, `Qwen2`, `Qwen3`, `Gemma`), and **Universal ViT / Multimodal Encoders** via declarative `ASIG0001` computation graphs.

Supports universal 8-format quantization (**`F32`**, **`F16`**, **`BF16`**, **`Q4_0`**, **`Q8_0`**, **`Q4_K`**, **`Q5_K`**, **`Q6_K`**) across both the **Sieve `1bit1024` two-stage `lm_head`** and **E05 v2 Grammar-Aware Sparse Head**.

## How It Compares to `llama.cpp`

| Component | `llama.cpp` Equivalent | `lila-asi` Component | Key Capabilities |
|---|---|---|---|
| **Model File** | `.gguf` | **`.asi` (`ASI v2` Cartridge)** | Page-aligned (`4096B`), zero-copy `mmap`, per-section **CRC-64** integrity verification, embedded tokenizer + chat template. |
| **Core C Library** | `libllama.so` + `llama.h` | **`lib/liblila_asi.so`** + `include/lila_asi.h` | Pure C11 with AVX2/FMA/F16C kernels across 8 quant formats, universal **Sieve `1bit1024` two-stage `lm_head`**, **E05 v2 Grammar-Aware Sparse Head** (up to 512 tools, optional args, RFC-8259 UTF-8 strings), and **E06 State-Centric Runtime** (16-slot Copy-on-Write State Tree `ASITREE1` + `SIGKILL`-recoverable Session WAL + confidence sampler `lila_asi_sample_ex`). |
| **CLI Tool** | `llama-cli` / `llama-bench` | **`bin/asi-cli`** & **`bin/lila-asi-cli`** | 1-command GGUF -> `.asi` packing, cartridge inspection, streaming prompt execution, interactive multi-turn chat, and native speed benchmarking. |
| **Python SDK** | `llama-cpp-python` | **`python/lila_asi`** (`from lila_asi import AsiRuntime`) | Pure Python stdlib `ctypes` wrapper (**zero pip dependencies**) exposing an OpenAI-compatible `.create_chat_completion(messages, stream=True)` API with automatic RAM + disk system-prompt prefix caching (`ASIPFX01`). |

---

## 1. Command-Line Quickstart (`asi-cli` & `lila-asi-cli`)

### A. Convert any `.gguf` (LLM or Vision `mmproj`) into `.asi` / `.asiv`
```bash
# Auto-detects LLM GGUF (-> model.asi) or vision mmproj GGUF (-> mmproj.asiv)
./bin/asi-cli pack /path/to/model.gguf -o model.asi
./bin/asi-cli pack /path/to/mmproj.gguf -o mmproj.asiv

# Or pack paired text + vision together in one shot:
./bin/asi-cli pack /path/to/model.gguf -o model.asi --mmproj /path/to/mmproj.gguf
```

### B. Inspect cartridge or `.asiv` vision sidecar headers, sections, and CRC-64
```bash
./bin/asi-cli info model.asi
./bin/asi-cli info mmproj.asiv
```

### C. Run streaming generation or interactive chat
```bash
# Single turn
./bin/asi-cli run model.asi -p "Explain recursion in two sentences." --mode fast

# Interactive multi-turn chat (reuses DeltaNet + KV prefix across turns)
./bin/asi-cli chat model.asi --ctx 8192 --mode fast

# Native C11 benchmark (zero Python overhead)
./bin/lila-asi-cli model.asi --bench -t 2 -n 64 --mode fast
```

---

## 2. Python Quickstart (Zero Third-Party Dependencies)

Install into any virtualenv with `pip install .` or add `python/` to `PYTHONPATH`:

```python
from lila_asi import AsiRuntime, apply_asi_mode_preset

apply_asi_mode_preset("fast")  # "fast" (default Sieve), "balanced", or "exact"
llm = AsiRuntime("model.asi", n_ctx=8192, n_threads=2)

# Non-streaming
resp = llm.create_chat_completion([
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is 17 * 23?"},
], max_tokens=128, temperature=0.0)
print(resp["choices"][0]["message"]["content"])

# Streaming (UTF-8 safe across multi-byte token boundaries)
for chunk in llm.create_chat_completion([
    {"role": "user", "content": "Write a haiku about systems programming."},
], stream=True):
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)

llm.close()
```

---

## 3. C11 API Quickstart (`include/lila_asi.h`)

```c
#include "lila_asi.h"
#include "asi_state_centric.h"
#include <stdio.h>

int main(void) {
    char err[256] = {0};
    LilaAsiModel *model = lila_asi_open("model.asi", err, sizeof(err));
    if (!model) { fprintf(stderr, "open failed: %s\n", err); return 1; }

    AsiV2ModelConfig cfg;
    lila_asi_model_info(model, &cfg);
    LilaAsiSession *sess = lila_asi_session_create(model, 4096, 2, err, sizeof(err));

    float *logits = calloc(cfg.vocab_size, sizeof(float));
    int32_t prompt[3] = {151644, 872, 198};
    lila_asi_eval_batch(sess, prompt, 3, 0, logits);

    /* Fork a Copy-on-Write checkpoint before speculative branching (E06) */
    uint32_t cp_id = 0;
    uint64_t rng = 0x123456789ABCDEF0ULL;
    lila_asi_cp_fork(sess, 3, rng, "branch_root", &cp_id);

    /* Sample with exact confidence & entropy metrics (E06) */
    AsiSampleStats stats;
    int32_t next_tok = lila_asi_sample_ex(logits, cfg.vocab_size, 0.0f, 20, 0.95f, &rng, &stats);
    printf("Sampled token=%d (margin=%.3f, entropy=%.3f)\n", next_tok, stats.margin, stats.entropy);

    free(logits);
    lila_asi_session_free(sess);
    lila_asi_close(model);
    return 0;
}
```

## 4. Building from C11 Source

```bash
cd csrc && make
```

## License

Developed and maintained by TicketLabs, part of the Harboria Labs technology portfolio.
Licensed under the GNU Affero General Public License v3.0 (`AGPL-3.0-only`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
