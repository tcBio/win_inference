#pragma once

#include "utils/result.hpp"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

namespace qwen::tokenizer {

/// Token ID type
using TokenId = int32_t;
using TokenIds = std::vector<TokenId>;

/// Special tokens for Qwen2.5
struct SpecialTokens {
    TokenId pad_token_id = 151643;
    TokenId eos_token_id = 151645;      // <|endoftext|>
    TokenId bos_token_id = 151643;
    TokenId im_start_id = 151644;       // <|im_start|>
    TokenId im_end_id = 151645;         // <|im_end|>

    // Additional special tokens
    std::string pad_token = "<|endoftext|>";
    std::string eos_token = "<|im_end|>";
    std::string bos_token = "<|endoftext|>";
    std::string im_start = "<|im_start|>";
    std::string im_end = "<|im_end|>";
};

/// Tokenizer configuration
struct TokenizerConfig {
    std::string vocab_file;             // Path to vocab.json
    std::string merges_file;            // Path to merges.txt
    SpecialTokens special_tokens;
    bool add_bos = false;               // Add BOS token at start
    bool add_eos = false;               // Add EOS token at end
};

/// Abstract tokenizer interface
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    /// Initialize tokenizer from config
    [[nodiscard]] virtual Result<void> initialize(const TokenizerConfig& config) = 0;

    /// Encode text to token IDs
    [[nodiscard]] virtual Result<TokenIds> encode(const std::string& text) const = 0;

    /// Encode with special token handling
    [[nodiscard]] virtual Result<TokenIds> encode(
        const std::string& text,
        bool add_special_tokens) const = 0;

    /// Decode token IDs to text
    [[nodiscard]] virtual Result<std::string> decode(const TokenIds& tokens) const = 0;

    /// Decode a single token
    [[nodiscard]] virtual Result<std::string> decode_token(TokenId token) const = 0;

    /// Get vocabulary size
    [[nodiscard]] virtual size_t vocab_size() const = 0;

    /// Get special tokens
    [[nodiscard]] virtual const SpecialTokens& special_tokens() const = 0;

    /// Check if token is a special token
    [[nodiscard]] virtual bool is_special_token(TokenId token) const = 0;

    /// Convert token to string (for debugging)
    [[nodiscard]] virtual std::string token_to_string(TokenId token) const = 0;
};

/// Create a BPE tokenizer for Qwen2.5
std::unique_ptr<ITokenizer> create_qwen_tokenizer();

/// Tokenizer that wraps an external tokenizer library (e.g., tiktoken, sentencepiece)
/// Use this if you have a pre-built tokenizer
std::unique_ptr<ITokenizer> create_external_tokenizer(const std::string& library_path);

}  // namespace qwen::tokenizer
