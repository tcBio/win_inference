#include "tokenizer/tokenizer.hpp"
#include "utils/logger.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <map>
#include <unordered_set>

namespace qwen::tokenizer {

/// Simple BPE tokenizer implementation for Qwen2.5
/// For production, consider using a well-tested tokenizer library
class QwenBPETokenizer : public ITokenizer {
public:
    QwenBPETokenizer() = default;

    Result<void> initialize(const TokenizerConfig& config) override {
        config_ = config;

        // Load vocabulary
        auto vocab_result = load_vocab(config.vocab_file);
        if (vocab_result.is_error()) {
            return vocab_result.error();
        }

        // Load merges
        auto merges_result = load_merges(config.merges_file);
        if (merges_result.is_error()) {
            return merges_result.error();
        }

        // Build reverse vocabulary for decoding
        for (const auto& [token, id] : vocab_) {
            id_to_token_[id] = token;
        }

        // Register special tokens
        register_special_tokens();

        initialized_ = true;
        log_info("tokenizer", "initialized", {
            {"vocab_size", vocab_.size()},
            {"merges_count", merges_.size()}
        });

        return Result<void>::success();
    }

    Result<TokenIds> encode(const std::string& text) const override {
        return encode(text, false);
    }

    Result<TokenIds> encode(const std::string& text, bool add_special_tokens) const override {
        if (!initialized_) {
            return Error::internal("Tokenizer not initialized");
        }

        TokenIds result;

        if (add_special_tokens && config_.add_bos) {
            result.push_back(config_.special_tokens.bos_token_id);
        }

        // Tokenize the text
        auto tokens = tokenize(text);
        result.insert(result.end(), tokens.begin(), tokens.end());

        if (add_special_tokens && config_.add_eos) {
            result.push_back(config_.special_tokens.eos_token_id);
        }

        return result;
    }

    Result<std::string> decode(const TokenIds& tokens) const override {
        if (!initialized_) {
            return Error::internal("Tokenizer not initialized");
        }

        std::string result;
        for (TokenId token : tokens) {
            auto decoded = decode_token(token);
            if (decoded.is_error()) {
                continue;  // Skip unknown tokens
            }
            result += decoded.value();
        }

        // Handle byte-level encoding
        result = decode_bytes(result);

        return result;
    }

    Result<std::string> decode_token(TokenId token) const override {
        auto it = id_to_token_.find(token);
        if (it == id_to_token_.end()) {
            return Error::invalid_request("Unknown token ID: " + std::to_string(token));
        }
        return it->second;
    }

    size_t vocab_size() const override {
        return vocab_.size();
    }

    const SpecialTokens& special_tokens() const override {
        return config_.special_tokens;
    }

    bool is_special_token(TokenId token) const override {
        return special_token_ids_.count(token) > 0;
    }

    std::string token_to_string(TokenId token) const override {
        auto it = id_to_token_.find(token);
        if (it == id_to_token_.end()) {
            return "<unk:" + std::to_string(token) + ">";
        }
        return it->second;
    }

private:
    Result<void> load_vocab(const std::string& path) {
        if (path.empty()) {
            // Use built-in minimal vocab for testing
            setup_minimal_vocab();
            return Result<void>::success();
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            return Error::internal("Failed to open vocab file: " + path);
        }

        try {
            nlohmann::json vocab_json;
            file >> vocab_json;

            for (auto& [token, id] : vocab_json.items()) {
                vocab_[token] = id.get<TokenId>();
            }
        } catch (const std::exception& e) {
            return Error::internal("Failed to parse vocab: " + std::string(e.what()));
        }

        return Result<void>::success();
    }

    Result<void> load_merges(const std::string& path) {
        if (path.empty()) {
            // No merges for minimal vocab
            return Result<void>::success();
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            return Error::internal("Failed to open merges file: " + path);
        }

        std::string line;
        int rank = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;

            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            merges_[{first, second}] = rank++;
        }

        return Result<void>::success();
    }

    void setup_minimal_vocab() {
        // Minimal vocabulary for testing
        // In production, load from vocab.json
        vocab_["<|endoftext|>"] = 151643;
        vocab_["<|im_start|>"] = 151644;
        vocab_["<|im_end|>"] = 151645;

        // Add basic ASCII characters
        for (int i = 0; i < 256; ++i) {
            std::string byte_token = "Ġ" + std::string(1, static_cast<char>(i));
            vocab_[byte_token] = i;
        }
    }

    void register_special_tokens() {
        special_token_ids_.insert(config_.special_tokens.pad_token_id);
        special_token_ids_.insert(config_.special_tokens.eos_token_id);
        special_token_ids_.insert(config_.special_tokens.bos_token_id);
        special_token_ids_.insert(config_.special_tokens.im_start_id);
        special_token_ids_.insert(config_.special_tokens.im_end_id);

        // Map special token strings
        special_token_map_[config_.special_tokens.im_start] = config_.special_tokens.im_start_id;
        special_token_map_[config_.special_tokens.im_end] = config_.special_tokens.im_end_id;
        special_token_map_[config_.special_tokens.pad_token] = config_.special_tokens.pad_token_id;
    }

    TokenIds tokenize(const std::string& text) const {
        TokenIds result;

        // First, handle special tokens
        std::string remaining = text;
        size_t pos = 0;

        while (pos < remaining.length()) {
            bool found_special = false;

            // Check for special tokens at current position
            for (const auto& [special_str, special_id] : special_token_map_) {
                if (remaining.compare(pos, special_str.length(), special_str) == 0) {
                    result.push_back(special_id);
                    pos += special_str.length();
                    found_special = true;
                    break;
                }
            }

            if (!found_special) {
                // Tokenize the next chunk until a special token
                size_t end = remaining.length();
                for (const auto& [special_str, _] : special_token_map_) {
                    size_t special_pos = remaining.find(special_str, pos);
                    if (special_pos != std::string::npos && special_pos < end) {
                        end = special_pos;
                    }
                }

                std::string chunk = remaining.substr(pos, end - pos);
                if (!chunk.empty()) {
                    auto chunk_tokens = tokenize_chunk(chunk);
                    result.insert(result.end(), chunk_tokens.begin(), chunk_tokens.end());
                }
                pos = end;
            }
        }

        return result;
    }

    TokenIds tokenize_chunk(const std::string& text) const {
        TokenIds result;

        // Simple byte-level fallback tokenization
        // In production, implement full BPE
        for (unsigned char c : text) {
            auto it = vocab_.find(std::string(1, c));
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                // Use byte value as token ID (fallback)
                result.push_back(static_cast<TokenId>(c));
            }
        }

        return result;
    }

    std::string decode_bytes(const std::string& text) const {
        // Handle byte-level encoding cleanup
        std::string result;
        for (size_t i = 0; i < text.length(); ++i) {
            if (text[i] == 'Ġ' && i + 1 < text.length()) {
                result += ' ';
            } else if (text[i] != 'Ġ') {
                result += text[i];
            }
        }
        return result;
    }

    TokenizerConfig config_;
    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merges_;
    std::unordered_set<TokenId> special_token_ids_;
    std::unordered_map<std::string, TokenId> special_token_map_;
    bool initialized_ = false;
};

std::unique_ptr<ITokenizer> create_qwen_tokenizer() {
    return std::make_unique<QwenBPETokenizer>();
}

std::unique_ptr<ITokenizer> create_external_tokenizer(const std::string& /*library_path*/) {
    // Placeholder for external tokenizer integration
    // Would load a shared library and wrap it
    return create_qwen_tokenizer();
}

}  // namespace qwen::tokenizer
