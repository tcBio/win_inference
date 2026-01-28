#include "api/types.hpp"

namespace qwen::api {

void to_json(nlohmann::json& j, const Message& m) {
    j = {{"role", role_to_string(m.role)}, {"content", m.content}};
    if (m.name) j["name"] = *m.name;
    if (m.tool_calls) j["tool_calls"] = *m.tool_calls;
    if (m.tool_call_id) j["tool_call_id"] = *m.tool_call_id;
}

void from_json(const nlohmann::json& j, Message& m) {
    m.role = string_to_role(j.at("role").get<std::string>());
    m.content = j.value("content", "");
    if (j.contains("name")) m.name = j["name"].get<std::string>();
    if (j.contains("tool_calls")) m.tool_calls = j["tool_calls"];
    if (j.contains("tool_call_id")) m.tool_call_id = j["tool_call_id"].get<std::string>();
}

void from_json(const nlohmann::json& j, Tool& t) {
    t.type = j.at("type").get<std::string>();
    t.function = j.at("function");
}

void to_json(nlohmann::json& j, const ChatCompletionRequest& r) {
    j = {{"model", r.model}, {"messages", nlohmann::json::array()}, {"stream", r.stream}};
    for (const auto& m : r.messages) {
        nlohmann::json mj;
        to_json(mj, m);
        j["messages"].push_back(mj);
    }
    if (r.temperature) j["temperature"] = *r.temperature;
    if (r.top_p) j["top_p"] = *r.top_p;
    if (r.max_tokens) j["max_tokens"] = *r.max_tokens;
    if (r.stop) j["stop"] = *r.stop;
    if (r.seed) j["seed"] = *r.seed;
}

void from_json(const nlohmann::json& j, ChatCompletionRequest& r) {
    r.model = j.at("model").get<std::string>();

    r.messages.clear();
    for (const auto& mj : j.at("messages")) {
        Message m;
        from_json(mj, m);
        r.messages.push_back(m);
    }

    r.temperature = j.contains("temperature") ?
        std::optional<double>(j["temperature"].get<double>()) : std::nullopt;
    r.top_p = j.contains("top_p") ?
        std::optional<double>(j["top_p"].get<double>()) : std::nullopt;
    r.max_tokens = j.contains("max_tokens") ?
        std::optional<int32_t>(j["max_tokens"].get<int32_t>()) : std::nullopt;

    if (j.contains("stop")) {
        if (j["stop"].is_string()) {
            r.stop = std::vector<std::string>{j["stop"].get<std::string>()};
        } else if (j["stop"].is_array()) {
            r.stop = j["stop"].get<std::vector<std::string>>();
        }
    }

    r.stream = j.value("stream", false);
    r.seed = j.contains("seed") ?
        std::optional<int64_t>(j["seed"].get<int64_t>()) : std::nullopt;

    if (j.contains("tools")) {
        r.tools = std::vector<Tool>{};
        for (const auto& tj : j["tools"]) {
            Tool t;
            from_json(tj, t);
            r.tools->push_back(t);
        }
    }

    r.presence_penalty = j.contains("presence_penalty") ?
        std::optional<double>(j["presence_penalty"].get<double>()) : std::nullopt;
    r.frequency_penalty = j.contains("frequency_penalty") ?
        std::optional<double>(j["frequency_penalty"].get<double>()) : std::nullopt;
    r.user = j.contains("user") ?
        std::optional<std::string>(j["user"].get<std::string>()) : std::nullopt;
}

void to_json(nlohmann::json& j, const Usage& u) {
    j = {
        {"prompt_tokens", u.prompt_tokens},
        {"completion_tokens", u.completion_tokens},
        {"total_tokens", u.total_tokens}
    };
}

void to_json(nlohmann::json& j, const Choice& c) {
    j = {
        {"index", c.index},
        {"message", nlohmann::json()}
    };
    to_json(j["message"], c.message);
    if (c.finish_reason != FinishReason::None) {
        j["finish_reason"] = finish_reason_to_string(c.finish_reason);
    } else {
        j["finish_reason"] = nullptr;
    }
}

void to_json(nlohmann::json& j, const ChatCompletionResponse& r) {
    j = {
        {"id", r.id},
        {"object", r.object},
        {"created", r.created},
        {"model", r.model},
        {"choices", nlohmann::json::array()}
    };
    for (const auto& c : r.choices) {
        nlohmann::json cj;
        to_json(cj, c);
        j["choices"].push_back(cj);
    }
    to_json(j["usage"], r.usage);
}

void to_json(nlohmann::json& j, const Delta& d) {
    j = nlohmann::json::object();
    if (d.role) j["role"] = role_to_string(*d.role);
    if (d.content) j["content"] = *d.content;
    if (d.tool_calls) j["tool_calls"] = *d.tool_calls;
}

void to_json(nlohmann::json& j, const StreamChoice& c) {
    j = {{"index", c.index}};
    to_json(j["delta"], c.delta);
    if (c.finish_reason) {
        j["finish_reason"] = finish_reason_to_string(*c.finish_reason);
    } else {
        j["finish_reason"] = nullptr;
    }
}

void to_json(nlohmann::json& j, const ChatCompletionChunk& c) {
    j = {
        {"id", c.id},
        {"object", c.object},
        {"created", c.created},
        {"model", c.model},
        {"choices", nlohmann::json::array()}
    };
    for (const auto& ch : c.choices) {
        nlohmann::json chj;
        to_json(chj, ch);
        j["choices"].push_back(chj);
    }
}

void to_json(nlohmann::json& j, const ErrorResponse& e) {
    j = {
        {"error", {
            {"type", e.type},
            {"code", e.code},
            {"message", e.message}
        }}
    };
    if (e.param) {
        j["error"]["param"] = *e.param;
    }
}

void to_json(nlohmann::json& j, const ModelInfo& m) {
    j = {
        {"id", m.id},
        {"object", m.object},
        {"created", m.created},
        {"owned_by", m.owned_by}
    };
}

}  // namespace qwen::api
