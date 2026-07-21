// llama_jni.cpp
// JNI bridge between com.yuvika.assistant.LocalLLMEngine and llama.cpp.
// Implements the 4 native methods declared in LocalLLMEngine.java:
//   nativeLoadModel, nativeFreeModel, nativeGenerate, nativeGenerateStreaming
//
// Built by .github/workflows/build-llama-jni.yml against a pinned llama.cpp
// commit fetched fresh at CI time (no vendored llama.cpp source in this repo).

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>

#include "llama.h"

#define TAG "YuvikaLlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

// Wraps a loaded model + its dedicated context. The jlong handle passed back
// to Java is a pointer to one of these, cast to long.
struct YuvikaLlamaSession {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
};

// One-time llama.cpp backend init. llama_backend_init() is safe to call once
// per process; guarded with a static flag since nativeLoadModel may be
// called more than once across the app's lifetime (reload after settings
// change, etc).
void ensureBackendInit() {
    static bool initialized = false;
    if (!initialized) {
        llama_backend_init();
        initialized = true;
    }
}

std::string jstringToStd(JNIEnv* env, jstring s) {
    if (s == nullptr) return std::string();
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(s, chars);
    return result;
}

// Runs one full generation loop (greedy/top-p/temp sampling via llama_sampler
// chain). If sink/onToken is non-null, streams each piece as it's produced;
// otherwise accumulates the full string and returns it at the end. Shared by
// nativeGenerate and nativeGenerateStreaming so behavior can't drift between
// the two call paths.
std::string runGeneration(
        YuvikaLlamaSession* session,
        const std::string& prompt,
        int maxNewTokens,
        float temperature,
        float topP,
        float repeatPenalty,
        JNIEnv* env,
        jobject sinkObj,
        jmethodID onTokenMethod) {

    llama_context* ctx = session->ctx;
    const llama_vocab* vocab = session->vocab;

    // Tokenize prompt (add BOS, allow special tokens like chat template markers).
    const int nPromptMax = (int) prompt.size() + 32;
    std::vector<llama_token> promptTokens(nPromptMax);
    int nPromptTokens = llama_tokenize(
            vocab, prompt.c_str(), (int32_t) prompt.size(),
            promptTokens.data(), nPromptMax, /*add_special=*/true, /*parse_special=*/true);
    if (nPromptTokens < 0) {
        promptTokens.resize(-nPromptTokens);
        nPromptTokens = llama_tokenize(
                vocab, prompt.c_str(), (int32_t) prompt.size(),
                promptTokens.data(), (int32_t) promptTokens.size(), true, true);
    }
    promptTokens.resize(nPromptTokens);

    // Feed the prompt through in one batch.
    llama_batch batch = llama_batch_get_one(promptTokens.data(), nPromptTokens);
    if (llama_decode(ctx, batch) != 0) {
        LOGE("llama_decode failed on prompt");
        return std::string();
    }

    // Build a sampler chain: temperature + top-p + repetition penalty, then
    // final distribution sample. This mirrors the topP/repeatPenalty knobs
    // LocalLLMEngine.java exposes as user settings.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            /*penalty_last_n=*/64, repeatPenalty, /*freq=*/0.0f, /*present=*/0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string output;
    output.reserve(256);
    int nCur = nPromptTokens;

    for (int i = 0; i < maxNewTokens; i++) {
        llama_token newToken = llama_sampler_sample(chain, ctx, -1);

        if (llama_vocab_is_eog(vocab, newToken)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, newToken, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            output += piece;
            if (sinkObj != nullptr && onTokenMethod != nullptr) {
                jstring jpiece = env->NewStringUTF(piece.c_str());
                env->CallVoidMethod(sinkObj, onTokenMethod, jpiece);
                env->DeleteLocalRef(jpiece);
            }
        }

        llama_batch nextBatch = llama_batch_get_one(&newToken, 1);
        if (llama_decode(ctx, nextBatch) != 0) {
            LOGE("llama_decode failed at step %d", i);
            break;
        }
        nCur++;

        // Stay inside the context window configured at load time.
        if (nCur >= llama_n_ctx(ctx)) {
            LOGI("Hit context window limit at %d tokens", nCur);
            break;
        }
    }

    llama_sampler_free(chain);
    return output;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_yuvika_assistant_LocalLLMEngine_nativeLoadModel(
        JNIEnv* env, jobject /*thiz*/, jstring modelPath, jint contextSize, jint threadCount) {

    ensureBackendInit();
    std::string path = jstringToStd(env, modelPath);
    LOGI("Loading model from %s (ctx=%d, threads=%d)", path.c_str(), contextSize, threadCount);

    llama_model_params modelParams = llama_model_default_params();
    // CPU-only inference — no GPU/NNAPI backend assumed present on-device.
    modelParams.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(path.c_str(), modelParams);
    if (model == nullptr) {
        LOGE("llama_model_load_from_file failed for %s", path.c_str());
        return 0;
    }

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = (uint32_t) (contextSize > 0 ? contextSize : 2048);
    ctxParams.n_threads = threadCount > 0 ? threadCount : 4;
    ctxParams.n_threads_batch = ctxParams.n_threads;

    llama_context* ctx = llama_init_from_model(model, ctxParams);
    if (ctx == nullptr) {
        LOGE("llama_init_from_model failed");
        llama_model_free(model);
        return 0;
    }

    auto* session = new YuvikaLlamaSession();
    session->model = model;
    session->ctx = ctx;
    session->vocab = llama_model_get_vocab(model);

    LOGI("Model loaded OK, handle=%p", (void*) session);
    return (jlong) session;
}

JNIEXPORT void JNICALL
Java_com_yuvika_assistant_LocalLLMEngine_nativeFreeModel(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {

    if (handle == 0) return;
    auto* session = reinterpret_cast<YuvikaLlamaSession*>(handle);
    if (session->ctx != nullptr) llama_free(session->ctx);
    if (session->model != nullptr) llama_model_free(session->model);
    delete session;
    LOGI("Model freed, handle=%p", (void*) handle);
}

JNIEXPORT jstring JNICALL
Java_com_yuvika_assistant_LocalLLMEngine_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/, jlong handle, jstring prompt, jint maxNewTokens,
        jfloat temperature, jfloat topP, jfloat repeatPenalty) {

    if (handle == 0) return env->NewStringUTF("");
    auto* session = reinterpret_cast<YuvikaLlamaSession*>(handle);
    std::string promptStr = jstringToStd(env, prompt);

    std::string result = runGeneration(
            session, promptStr, maxNewTokens, temperature, topP, repeatPenalty,
            env, nullptr, nullptr);

    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_yuvika_assistant_LocalLLMEngine_nativeGenerateStreaming(
        JNIEnv* env, jobject /*thiz*/, jlong handle, jstring prompt, jint maxNewTokens,
        jfloat temperature, jfloat topP, jfloat repeatPenalty, jobject sink) {

    if (handle == 0 || sink == nullptr) return;
    auto* session = reinterpret_cast<YuvikaLlamaSession*>(handle);
    std::string promptStr = jstringToStd(env, prompt);

    jclass sinkClass = env->GetObjectClass(sink);
    jmethodID onTokenMethod = env->GetMethodID(sinkClass, "onNativeToken", "(Ljava/lang/String;)V");
    if (onTokenMethod == nullptr) {
        LOGE("Could not find onNativeToken(String) on sink object");
        env->DeleteLocalRef(sinkClass);
        return;
    }

    runGeneration(
            session, promptStr, maxNewTokens, temperature, topP, repeatPenalty,
            env, sink, onTokenMethod);

    env->DeleteLocalRef(sinkClass);
}

} // extern "C"
