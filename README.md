# possum [![Build and test](https://github.com/shayne-fletcher/possum/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/shayne-fletcher/possum/actions/workflows/build-and-test.yml)

[rustdoc docs](https://shayne-fletcher.github.io/possum/possum)
```
    Usage: possum [COMMAND]

    Commands:
      model  Do things with 🤗 models
      help   Print this message or the help of the given subcommand(s)

    Options:
      -h, --help     Print help
      -V, --version  Print version
```

### Downloading model files

Pick exactly what you need with `--include`/`--exclude` globs and bound the
parallelism with `--concurrency` (default 4):

```
# a DeepSeek model's weights, skipping the repo's figures
possum model download \
  --repository deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --include '*.safetensors' '*.json' --exclude 'figures/*'

# a single GGUF quant from a community repo
possum model download \
  --repository bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  --include '*Q4_K_M.gguf'
```

<img src="./img/possum.jpeg" alt="possum" style="float: left; width: 50%; height: 50%;">
