(* config.mli - Configuration management *)

open Types

(* ============================================================================
   CONFIGURATION LOADING
   ============================================================================ *)

(* Load configuration from file (JSON or YAML) *)
val load_config : string -> (config, string) result

(* Create default configuration *)
val default_config : unit -> config

(* Save configuration to file *)
val save_config : config -> string -> (unit, string) result


(* ============================================================================
   PRESET CONFIGURATIONS
   ============================================================================ *)

(* Small/fast config for testing *)
val test_config : config

(* Config matching Jacob Hilton's setup *)
val jacob_hilton_config : config

(* Medium baseline config *)
val baseline_config : config

(* Large-scale training config *)
val production_config : config

(* GPU-optimized config *)
val gpu_optimized_config : config


(* ============================================================================
   CONFIGURATION BUILDERS
   ============================================================================ *)

(* Builder pattern for constructing configs *)
type config_builder

val create_builder : unit -> config_builder

(* Transformer architecture *)
val with_num_layers : config_builder -> int -> config_builder
val with_embed_dim : config_builder -> int -> config_builder
val with_num_heads : config_builder -> int -> config_builder
val with_ff_dim : config_builder -> int -> config_builder

(* Training *)
val with_learning_rate : config_builder -> float -> config_builder
val with_batch_size : config_builder -> int -> config_builder
val with_replay_buffer_size : config_builder -> int -> config_builder

(* Encoding *)
val with_encoding_type : config_builder -> string -> config_builder
val with_feature_dim : config_builder -> int -> config_builder

(* System *)
val with_device : config_builder -> string -> config_builder
val with_seed : config_builder -> int -> config_builder

(* Build final config *)
val build : config_builder -> config


(* ============================================================================
   CONFIGURATION VALIDATION
   ============================================================================ *)

(* Validate that configuration is sensible *)
val validate_config : config -> (bool, string list) result

(* Check if config is compatible with hardware *)
val check_hardware_compatibility : config -> (bool, string) result

(* Estimate memory requirements *)
val estimate_memory_usage : config -> int  (* bytes *)

(* Estimate training time *)
val estimate_training_time : config -> float  (* hours *)


(* ============================================================================
   CONFIGURATION UTILITIES
   ============================================================================ *)

(* Print config in human-readable format *)
val print_config : config -> unit

(* Convert config to JSON string *)
val config_to_json : config -> string

(* Parse config from JSON string *)
val config_from_json : string -> (config, string) result

(* Merge two configs (second overrides first) *)
val merge_configs : config -> config -> config

(* Override config with command-line arguments *)
val override_from_args : config -> string array -> config


(* ============================================================================
   HYPERPARAMETER PRESETS
   ============================================================================ *)

(* Learning rate schedules *)
val constant_lr : float -> (int -> float)
val linear_warmup_lr : float -> int -> (int -> float)
val cosine_decay_lr : float -> int -> (int -> float)
val exponential_decay_lr : float -> float -> (int -> float)


(* ============================================================================
   EXPERIMENT TRACKING
   ============================================================================ *)

(* Generate unique experiment ID *)
val generate_experiment_id : config -> string

(* Create experiment directory structure *)
val setup_experiment_dir : config -> string -> (unit, string) result

(* Log configuration to experiment tracker *)
val log_config : config -> unit


(* ============================================================================
   CONFIGURATION SEARCH
   ============================================================================ *)

(* Grid search over hyperparameters *)
val grid_search_configs : config -> string list -> 'a list list -> config list

(* Random search *)
val random_search_configs : config -> int -> rng_state -> config list * rng_state
