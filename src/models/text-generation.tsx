import "text-encoding-polyfill";
import { Tensor } from "onnxruntime-react-native";
import { Base } from "./base";

const batchSize = 32;

/**
 * Class to handle a large language model on top of onnxruntime
 */
export class TextGeneration extends Base {
  public outputTokens: bigint[] = [];
  private needPositionIds = true;
  private stopGeneration = false;
  private attentionMaskTensor: Tensor | null = null;

  public initializeFeed() {
    super.initializeFeed();
    this.outputTokens = [];
  }

  public async generate(
    tokens: bigint[],
    callback: (tokens: bigint[]) => void,
    options: { maxTokens: number },
  ): Promise<bigint[]> {
    const maxTokens = options.maxTokens;
    const feed = this.feed;

    // Pre-allocate reusable tensors
    const singleTokenTensor = new Tensor("int64", [0], [1, 1]); // Initialize with safe value
    const attentionMaskBuffer = new BigInt64Array(maxTokens);
    attentionMaskBuffer.fill(1n);

    // Safely convert tokens to BigInt64Array
    const initialTokens = new BigInt64Array(tokens.length);
    tokens.forEach((token, i) => {
      initialTokens[i] = BigInt(Number(token)); // Ensure safe conversion
    });

    const inputIdsTensor = new Tensor("int64", initialTokens, [
      1,
      tokens.length,
    ]);
    feed.input_ids = inputIdsTensor;

    this.stopGeneration = false;
    this.outputTokens = [...tokens]; // Safe copy

    let lastToken = 0n;
    let sequenceLength = this.outputTokens.length;
    const initialLength = feed.input_ids.size;

    // Pre-allocate position IDs tensor if needed
    let positionIdsTensor: Tensor | null = null;
    if (this.needPositionIds) {
      const positionIds = new BigInt64Array(initialLength);
      for (let i = 0; i < initialLength; i++) {
        positionIds[i] = BigInt(sequenceLength - initialLength + i);
      }
      positionIdsTensor = new Tensor("int64", positionIds, [1, initialLength]);
      feed.position_ids = positionIdsTensor;
    }

    if (!this.sess) {
      throw new Error("Session is undefined");
    }

    try {
      while (
        lastToken !== this.eos &&
        lastToken !== 32007n &&
        sequenceLength < maxTokens &&
        !this.stopGeneration
      ) {
        sequenceLength = this.outputTokens.length;

        // Create attention mask using BigInt64Array
        const attentionMask = new BigInt64Array(sequenceLength);
        attentionMask.fill(1n);
        const newAttentionMask = new Tensor("int64", attentionMask, [
          1,
          sequenceLength,
        ]);

        if (this.attentionMaskTensor) {
          this.attentionMaskTensor.dispose();
        }

        this.attentionMaskTensor = newAttentionMask;
        feed.attention_mask = this.attentionMaskTensor;

        const outputs = await this.sess.run(feed, {
          extra: {
            memory: {
              enable_memory_arena_shrinkage: "cpu:0",
              arena_extend_strategy: "kNextPowerOfTwo",
            },
          },
        });

        // Safely handle logits output
        const logitsOutput = outputs.logits;
        if (!logitsOutput) {
          throw new Error("No logits in model output");
        }

        try {
          lastToken = BigInt(Math.floor(this.argmax(logitsOutput)));
          this.outputTokens.push(lastToken);
        } catch (e) {
          console.error("Token conversion error:", e);
          break;
        }

        if (callback && this.outputTokens.length % batchSize === 0) {
          callback(this.outputTokens);
        }

        this.updateKVCache(feed, outputs);

        // Safe token tensor update
        const newTokenData = new BigInt64Array([lastToken]);
        const newSingleToken = new Tensor("int64", newTokenData, [1, 1]);
        singleTokenTensor.dispose();
        feed.input_ids = newSingleToken;

        if (this.needPositionIds) {
          const newPosData = new BigInt64Array([BigInt(sequenceLength)]);
          const newPositionIds = new Tensor("int64", newPosData, [1, 1]);

          if (positionIdsTensor) {
            positionIdsTensor.dispose();
          }

          positionIdsTensor = newPositionIds;
          feed.position_ids = positionIdsTensor;
        }
      }

      if (callback) {
        callback(this.outputTokens); // Final callback with safe copy
      }

      return this.outputTokens;
    } finally {
      inputIdsTensor.dispose();
      this.attentionMaskTensor?.dispose();
      positionIdsTensor?.dispose();
    }
  }

  /**
   * Cleanup resources
   */
  public dispose(): void {
    this.outputTokens = [];

    if (this.attentionMaskTensor) {
      this.attentionMaskTensor.dispose();
      this.attentionMaskTensor = null;
    }

    this.release();
  }

  /**
   * Stop the generation process
   */
  public stop(): void {
    this.stopGeneration = true;
  }
}
