use std::io::Read;
use std::io::Write;
use clap::ValueEnum;

use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use tokenizers::tokenizer::Tokenizer;


use varnish::run_vtc_tests;
use varnish::vcl::{Backend, Ctx, Serve, Transfer, VclResult};

run_vtc_tests!("tests/*.vtc");

#[varnish::vmod]
mod candletest {
    use std::error::Error;

    use varnish::vcl::{Backend, Ctx};
    use varnish_sys::ffi::VCL_BACKEND;

    use super::root;
    use crate::new_model;

    // Rust implementation of the VCC object, it mirrors what happens in C, except
    // for a couple of points:
    // - we create and return a Rust object, instead of a void pointer
    // - new() returns a Result, leaving the error handling to varnish-rs
    impl root {
        pub fn new(
            ctx: &mut Ctx,
            #[vcl_name] vcl_name: &str,
        ) -> Result<Self, Box<dyn Error>> {

            // store the mime database in memory, possibly
            let backend = Backend::new(
                ctx,
                vcl_name,
                new_model(),
                false,
            )?;
            Ok(root { backend })
        }

        pub fn backend(&self, _ctx: &Ctx) -> VCL_BACKEND {
            self.backend.vcl_ptr()
        }
    }
}

// root is the Rust implement of the VCC definition (in vmod.vcc)
// it only contains backend, which wraps a FileBackend, and
// handles response body creation with a FileTransfer
#[allow(non_camel_case_types)]
struct root {
    backend: Backend<Model, FileTransfer>,
}

impl Serve<FileTransfer> for Model {
    fn get_type(&self) -> &str {
        "candletest"
    }

    fn get_headers(&self, ctx: &mut Ctx) -> VclResult<Option<FileTransfer>> {
        // let's start building our response
        let beresp = ctx.http_beresp.as_mut().unwrap();
        beresp.set_proto("HTTP/1.1")?;

        let bereq = ctx.http_bereq.as_ref().unwrap();
        let answer = self.ask(bereq.header("ask"));

        let mut transfer = None;
        beresp.set_status(200);
        if bereq.method() == Some("GET") {
            transfer = Some(FileTransfer {
                // prevent reading more than expected
                reader: std::io::Cursor::new(answer),
            });
        }
        Ok(transfer)
    }
}

struct FileTransfer {
    reader: std::io::Cursor<Vec<u8>>,
}

impl Transfer for FileTransfer {
    fn read(&mut self, buf: &mut [u8]) -> VclResult<usize> {
        self.reader.read(buf).map_err(|e| e.to_string().into())
    }
    fn len(&self) -> Option<usize> {
        None
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "7b")]
    L7b,
    #[value(name = "13b")]
    L13b,
    #[value(name = "70b")]
    L70b,
    #[value(name = "7b-chat")]
    L7bChat,
    #[value(name = "13b-chat")]
    L13bChat,
    #[value(name = "70b-chat")]
    L70bChat,
    #[value(name = "7b-code")]
    L7bCode,
    #[value(name = "13b-code")]
    L13bCode,
    #[value(name = "32b-code")]
    L34bCode,
    #[value(name = "7b-leo")]
    Leo7b,
    #[value(name = "13b-leo")]
    Leo13b,
    #[value(name = "7b-mistral")]
    Mistral7b,
    #[value(name = "7b-mistral-instruct")]
    Mistral7bInstruct,
    #[value(name = "7b-mistral-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7b-zephyr-a")]
    Zephyr7bAlpha,
    #[value(name = "7b-zephyr-b")]
    Zephyr7bBeta,
    #[value(name = "7b-open-chat-3.5")]
    OpenChat35,
    #[value(name = "7b-starling-a")]
    Starling7bAlpha,
    #[value(name = "mixtral")]
    Mixtral,
    #[value(name = "mixtral-instruct")]
    MixtralInstruct,
    #[value(name = "llama3-8b")]
    L8b,
    #[value(name = "phi3")]
    Phi3,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-1.7B-Instruct")]
    SmolLM2_1BInstruct,
}

impl Which {
    fn is_mistral(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::L8b
            | Self::Phi3
            | Self::SmolLM2_1BInstruct
            | Self::SmolLM2_360MInstruct => false,
            // Zephyr and OpenChat are fine tuned versions of mistral and should be treated in the
            // same way. Starling is a fine tuned version of OpenChat.
            Self::OpenChat35
            | Self::Starling7bAlpha
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02 => true,
        }
    }

    fn is_zephyr(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::OpenChat35
            | Self::Starling7bAlpha
            | Self::L8b
            | Self::SmolLM2_1BInstruct
            | Self::SmolLM2_360MInstruct
            | Self::Phi3 => false,
            Self::Zephyr7bAlpha | Self::Zephyr7bBeta => true,
        }
    }

    fn is_open_chat(&self) -> bool {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode
            | Self::Leo7b
            | Self::Leo13b
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta
            | Self::L8b
            | Self::SmolLM2_1BInstruct
            | Self::SmolLM2_360MInstruct
            | Self::Phi3 => false,
            Self::OpenChat35 | Self::Starling7bAlpha => true,
        }
    }

    fn tokenizer_repo(&self) -> &'static str {
        match self {
            Self::L7b
            | Self::L13b
            | Self::L70b
            | Self::L7bChat
            | Self::L13bChat
            | Self::L70bChat
            | Self::L7bCode
            | Self::L13bCode
            | Self::L34bCode => "hf-internal-testing/llama-tokenizer",
            Self::Leo7b => "LeoLM/leo-hessianai-7b",
            Self::Leo13b => "LeoLM/leo-hessianai-13b",
            Self::Mixtral => "mistralai/Mixtral-8x7B-v0.1",
            Self::MixtralInstruct => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructV02
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta => "mistralai/Mistral-7B-v0.1",
            Self::OpenChat35 => "openchat/openchat_3.5",
            Self::Starling7bAlpha => "berkeley-nest/Starling-LM-7B-alpha",
            Self::L8b => "meta-llama/Meta-Llama-3-8B",
            Self::Phi3 => "microsoft/Phi-3-mini-4k-instruct",
            Self::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
            Self::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        }
    }
}

struct Args {
    model: Option<String>,
    prompt: Option<String>,
    sample_len: usize,
    tokenizer: Option<String>,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: u64,
    verbose_prompt: bool,
    split_prompt: bool,
    cpu: bool,
    repeat_penalty: f32,
    repeat_last_n: usize,
    which: Which,
    gqa: Option<usize>,
    force_dmmv: bool,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let model_path = match &self.model {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let (repo, filename) = match self.which {
                    Which::L7b => ("TheBloke/Llama-2-7B-GGML", "llama-2-7b.ggmlv3.q4_0.bin"),
                    Which::L13b => ("TheBloke/Llama-2-13B-GGML", "llama-2-13b.ggmlv3.q4_0.bin"),
                    Which::L70b => ("TheBloke/Llama-2-70B-GGML", "llama-2-70b.ggmlv3.q4_0.bin"),
                    Which::L7bChat => (
                        "TheBloke/Llama-2-7B-Chat-GGML",
                        "llama-2-7b-chat.ggmlv3.q4_0.bin",
                    ),
                    Which::L13bChat => (
                        "TheBloke/Llama-2-13B-Chat-GGML",
                        "llama-2-13b-chat.ggmlv3.q4_0.bin",
                    ),
                    Which::L70bChat => (
                        "TheBloke/Llama-2-70B-Chat-GGML",
                        "llama-2-70b-chat.ggmlv3.q4_0.bin",
                    ),
                    Which::L7bCode => ("TheBloke/CodeLlama-7B-GGUF", "codellama-7b.Q8_0.gguf"),
                    Which::L13bCode => ("TheBloke/CodeLlama-13B-GGUF", "codellama-13b.Q8_0.gguf"),
                    Which::L34bCode => ("TheBloke/CodeLlama-34B-GGUF", "codellama-34b.Q8_0.gguf"),
                    Which::Leo7b => (
                        "TheBloke/leo-hessianai-7B-GGUF",
                        "leo-hessianai-7b.Q4_K_M.gguf",
                    ),
                    Which::Leo13b => (
                        "TheBloke/leo-hessianai-13B-GGUF",
                        "leo-hessianai-13b.Q4_K_M.gguf",
                    ),
                    Which::Mixtral => (
                        "TheBloke/Mixtral-8x7B-v0.1-GGUF",
                        "mixtral-8x7b-v0.1.Q4_K_M.gguf",
                    ),
                    Which::MixtralInstruct => (
                        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                        "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
                    ),
                    Which::Mistral7b => (
                        "TheBloke/Mistral-7B-v0.1-GGUF",
                        "mistral-7b-v0.1.Q4_K_S.gguf",
                    ),
                    Which::Mistral7bInstruct => (
                        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
                    ),
                    Which::Mistral7bInstructV02 => (
                        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                        "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
                    ),
                    Which::Zephyr7bAlpha => (
                        "TheBloke/zephyr-7B-alpha-GGUF",
                        "zephyr-7b-alpha.Q4_K_M.gguf",
                    ),
                    Which::Zephyr7bBeta => {
                        ("TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf")
                    }
                    Which::OpenChat35 => ("TheBloke/openchat_3.5-GGUF", "openchat_3.5.Q4_K_M.gguf"),
                    Which::Starling7bAlpha => (
                        "TheBloke/Starling-LM-7B-alpha-GGUF",
                        "starling-lm-7b-alpha.Q4_K_M.gguf",
                    ),
                    // TODO: swap to TheBloke model when available
                    Which::L8b => (
                        "QuantFactory/Meta-Llama-3-8B-GGUF",
                        "Meta-Llama-3-8B.Q4_K_S.gguf",
                    ),
                    Which::Phi3 => (
                        "microsoft/Phi-3-mini-4k-instruct-gguf",
                        "Phi-3-mini-4k-instruct-q4.gguf",
                    ),
                    Which::SmolLM2_360MInstruct => (
                        "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
                        "smollm2-360m-instruct-q8_0.gguf",
                    ),
                    Which::SmolLM2_1BInstruct => (
                        "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
                        "smollm2-1.7b-instruct-q4_k_m.gguf",
                    ),
                };
                let revision = if self.which == Which::Phi3 {
                    "5eef2ce24766d31909c0b269fe90c817a8f263fb"
                } else {
                    "main"
                };
                let api = hf_hub::api::sync::Api::new()?;
                let repo = api.repo(hf_hub::Repo::with_revision(
                    repo.to_string(),
                    hf_hub::RepoType::Model,
                    revision.to_string(),
                ));
                repo.get(filename)?
            }
        };
        Ok(model_path)
    }
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

struct Model {
    model: ModelWeights,
    args: Args,
    device: Device,
}

fn new_model() -> Model {
    let args = Args{
        model: None,

        sample_len: 50,
        temperature: 0.8,
        top_p: None,
        tokenizer: None,
        prompt: Some("Tell me about Paris".to_string()),
        top_k: None,
        seed: 299792458,
        verbose_prompt: false,
        split_prompt: false,
        cpu: false,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        which: Which::L7b,
        gqa: None,
        force_dmmv: false,
    };
    let model_path = args.model().unwrap();
    let mut file = std::fs::File::open(&model_path).unwrap();
    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu).unwrap();

    let model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path)).unwrap();
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            ModelWeights::from_gguf(model, &mut file, &device).unwrap()
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let model = ggml_file::Content::read(&mut file, &device)
                .map_err(|e| e.with_path(model_path)).unwrap();
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensors.iter() {
                let elem_count = tensor.shape().elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.dtype().type_size() / tensor.dtype().block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensors.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            println!("params: {:?}", model.hparams);
            let default_gqa = match args.which {
                Which::L7b
                | Which::L13b
                | Which::L7bChat
                | Which::L13bChat
                | Which::L7bCode
                | Which::L13bCode
                | Which::L34bCode
                | Which::Leo7b
                | Which::Leo13b
                | Which::L8b
                | Which::SmolLM2_1BInstruct
                | Which::SmolLM2_360MInstruct
                | Which::Phi3 => 1,
                Which::Mixtral
                | Which::MixtralInstruct
                | Which::Mistral7b
                | Which::Mistral7bInstruct
                | Which::Mistral7bInstructV02
                | Which::Zephyr7bAlpha
                | Which::Zephyr7bBeta
                | Which::L70b
                | Which::L70bChat
                | Which::OpenChat35
                | Which::Starling7bAlpha => 8,
            };
            ModelWeights::from_ggml(model, args.gqa.unwrap_or(default_gqa)).unwrap()
        }
    };

    Model {
        model,
        args,
        device,
    }
}

impl Model {
    fn ask(&self, prompt: Option<&str>) -> Vec<u8> {
        let mut answer = Vec::new();
        let mut model = self.model.clone();

        let tokenizer =         {
                let api = hf_hub::api::sync::Api::new().unwrap();
                let repo = Which::L7b.tokenizer_repo();
                let api = api.model(repo.to_string());
                Tokenizer::from_file(api.get("tokenizer.json").unwrap()).unwrap()
            }
;
        let mut tos = TokenOutputStream::new(tokenizer);

        let pre_prompt_tokens = vec![];
        let prompt_str = prompt.unwrap_or("Tell me about Paris");

        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg).unwrap();
        if self.args.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = self.args.sample_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let mut logits_processor = {
            let temperature = self.args.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (self.args.top_k, self.args.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(self.args.seed, sampling)
        };

        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = if !self.args.split_prompt {
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device).unwrap().unsqueeze(0).unwrap();
            let logits = model.forward(&input, 0).unwrap();
            let logits = logits.squeeze(0).unwrap();
            logits_processor.sample(&logits).unwrap()
        } else {
            let mut next_token = 0;
            for (pos, token) in prompt_tokens.iter().enumerate() {
                let input = Tensor::new(&[*token], &self.device).unwrap().unsqueeze(0).unwrap();
                let logits = model.forward(&input, pos).unwrap();
                let logits = logits.squeeze(0).unwrap();
                next_token = logits_processor.sample(&logits).unwrap()
            }
            next_token
        };
        let prompt_dt = start_prompt_processing.elapsed();
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token).unwrap() {
            print!("{t}");
            std::io::stdout().flush().unwrap();
        }

        let eos_token = match self.args.which {
            Which::SmolLM2_360MInstruct | Which::SmolLM2_1BInstruct => "<|endoftext|>",
            Which::L8b => "<|end_of_text|>",
            _ => match self.args.which.is_open_chat() {
                true => "<|end_of_turn|>",
                false => "</s>",
            },
        };

        let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device).unwrap().unsqueeze(0).unwrap();
            let logits = model.forward(&input, prompt_tokens.len() + index).unwrap();
            let logits = logits.squeeze(0).unwrap();
            let logits = if self.args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.args.repeat_penalty,
                    &all_tokens[start_at..],
                ).unwrap()
            };
            next_token = logits_processor.sample(&logits).unwrap();
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token).unwrap() {
                print!("{t}");
                for c in t.as_bytes() {
                    answer.push(*c);
                }
                std::io::stdout().flush().unwrap();
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg).unwrap() {
            print!("{rest}");
        }
        std::io::stdout().flush().unwrap();
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );

        answer
    }
}
