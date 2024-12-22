import "text-encoding-polyfill";
import { env, InferenceSession, Tensor } from "onnxruntime-react-native";

declare global {
  var gc: () => void;
}

async function load(uri: string): Promise<ArrayBuffer> {
  // @ts-ignore
  return await fetch(uri).then((response) => response.arrayBuffer());
}

function getHuggingfaceUrl(model: string, filepath: string): string {
  return `https://huggingface.co/${model}/${filepath}`;
}

/** Load Options */
export interface LoadOptions {
  /** The maximum number of tokens for text generation. */
  max_tokens: number;
  /** Enables verbose logging. */
  verbose: boolean;
  /** Indicates if external data is used. */
  externalData: boolean;
  dataFileName?: string;
  /** Function to fetch external data. */
  fetch: (url: string) => Promise<string>;
  /** List of execution providers for ONNX runtime. */
  executionProviders: InferenceSession.ExecutionProviderConfig[];
}

export class Base {
  protected sess?: InferenceSession;
  protected feed: Record<string, Tensor> = Object.create(null);
  protected eos = 2n;
  private kv_dims: number[] = [];
  private num_layers = 0;
  private dtype: "float16" | "float32" = "float32";

  // biome-ignore lint/complexity/noUselessConstructor: <explanation>
  constructor() {}

  async load(
    model: string,
    onnx_file = "onnx/model.onnx",
    options: LoadOptions,
  ) {
    const verbose = options.verbose;
    const fetch = options.fetch;

    try {
      const json_bytes = await load(
        await fetch(getHuggingfaceUrl(model, "config.json")),
      );
      // @ts-ignore
      const textDecoder = new TextDecoder();
      const model_config = JSON.parse(textDecoder.decode(json_bytes));
      const model_path = await fetch(getHuggingfaceUrl(model, onnx_file));

      // Rest of the code remains the same
      const opt: InferenceSession.SessionOptions = {
        executionProviders: options.executionProviders,
        graphOptimizationLevel: "all",
      };

      if (options.externalData) {
        opt.externalData = [
          await fetch(
            getHuggingfaceUrl(
              model,
              options.dataFileName ?? `${onnx_file}_data`,
            ),
          ),
        ];
      }

      if (verbose) {
        opt.logSeverityLevel = 0;
        opt.logVerbosityLevel = 0;
        env.logLevel = "verbose";
      }

      // Release existing session if any
      await this.release();

      this.sess = await InferenceSession.create(model_path, opt);
      this.eos = BigInt(model_config.eos_token_id);
      this.kv_dims = [
        1,
        model_config.num_key_value_heads,
        0,
        model_config.hidden_size / model_config.num_attention_heads,
      ];
      this.num_layers = model_config.num_hidden_layers;
      this.initializeFeed();
    } catch (error) {
      await this.release();
      throw error;
    }
  }

  public initializeFeed() {
    // Preallocate capacity for better memory usage
    const feedSize = this.num_layers * 2;
    const feed: Record<string, Tensor> = Object.create(null);

    // Batch dispose existing tensors
    const tensorsToDispose = Object.values(this.feed).filter(
      (tensor) => tensor?.location === "gpu-buffer",
    );

    // biome-ignore lint/complexity/noForEach: <explanation>
    tensorsToDispose.forEach((tensor) => tensor.dispose());

    // Reuse array allocation
    const empty =
      this.dtype === "float16" ? new Uint16Array(0) : new Float32Array(0);

    // Create tensors in a single loop
    for (let i = 0; i < this.num_layers; i++) {
      feed[`past_key_values.${i}.key`] = new Tensor(
        this.dtype,
        empty,
        this.kv_dims,
      );
      feed[`past_key_values.${i}.value`] = new Tensor(
        this.dtype,
        empty,
        this.kv_dims,
      );
    }

    this.feed = feed;
  }

  protected argmax(t: Tensor): number {
    const arr = t.data;
    const start = t.dims[2] * (t.dims[1] - 1);
    let max = arr[start];
    let maxidx = 0;

    for (let i = 0; i < t.dims[2]; i++) {
      const val = arr[i + start];
      if (!Number.isFinite(val as number)) {
        throw new Error("found infinitive in logits");
      }
      if (val > max) {
        max = val;
        maxidx = i;
      }
    }
    return maxidx;
  }

  protected updateKVCache(
    feed: Record<string, Tensor>,
    outputs: InferenceSession.OnnxValueMapType,
  ) {
    // Cleanup existing tensors first
    for (const name in feed) {
      if (name.startsWith("past_key_values")) {
        const tensor = feed[name];
        if (tensor?.location === "gpu-buffer") {
          tensor.dispose();
        }
      }
    }

    // Update with new tensors
    for (const name in outputs) {
      if (name.startsWith("present")) {
        const newName = name.replace("present", "past_key_values");
        feed[newName] = outputs[name];
      }
    }
  }

  async release() {
    if (this.sess) {
      // Cleanup feed tensors first
      // biome-ignore lint/complexity/noForEach: <explanation>
      Object.values(this.feed).forEach((tensor) => {
        if (tensor?.location === "gpu-buffer") {
          tensor.dispose();
        }
      });

      // Clear feed reference
      this.feed = Object.create(null);

      // Release session
      await this.sess.release();
      this.sess = undefined;

      // Suggest garbage collection
      if (typeof gc !== "undefined") {
        gc();
      }
    }
  }
}
