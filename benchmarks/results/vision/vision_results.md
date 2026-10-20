# Vision test results (2026-10-08)

Server: Qwen3.6-35B-A3B UD-IQ4_XS + `mmproj-F16.gguf`, projector on the CPU (`--no-mmproj-offload`), 64k context, thinking off, temperature 0. Requests went through the gateway in OpenAI format (`image_url` with a base64 data URL), exactly as Cline sends them.

Test images (synthetic, known content): [`test-invoice-chart.png`](../../../assets/vision/test-invoice-chart.png) (800×500) and [`test-code-screenshot-1080p.png`](../../../assets/vision/test-code-screenshot-1080p.png) (1920×1080).

## Speed

| Image | `--image-max-tokens` | Prompt tokens (image + text) | Prompt processing (incl. CPU image encoding) | Total wall time | Generation |
|---|---|---|---|---|---|
| 800×500 | default | 447 | 34.3 s | 50.9 s | 15.3 tok/s |
| 1920×1080 | default | 2,082 | 140.3 s | 147.9 s | 16.4 tok/s |
| 800×500 | **1024** | 437 | 19.2 s | 22.1 s | — |
| 1920×1080 | **1024** | **1,040** | **68.3 s** | **76.9 s** | — |

Memory with vision loaded: VRAM 9.5–9.7 GB (unchanged vs. text-only), server RSS up to ~11.3 GB, RAM available 10.7 GB, swap flat.

## Accuracy

**Invoice + chart image.** Asked for invoice number, customer, total, shapes and colours, and the day with the most orders:

> Invoice Number: #4817 · Customer Name: Ravi Kumar · Total: $23.50 · Shapes with Colours: Red Circle, Blue Square · Day with Most Orders: Tuesday (7 orders)

All 5 facts correct (default run also read the bar values 3 / 7 / 5 and noted the green text).

**1080p code screenshot, transcription** (cap 1024): character-exact transcription of all 13 lines, including the comment.

**1080p code screenshot, reasoning** (default): the model read the code correctly, but **trusted a misleading planted comment** (`# TypeError: 'tuple' object does not support item assignment`) and reported a bug that does not exist (`out[-1] = (...)` assigns a list element, which is valid). Lesson: image *reading* is reliable; reasoning about the content still needs review, just as with text.
