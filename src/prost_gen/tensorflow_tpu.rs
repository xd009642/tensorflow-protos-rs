/// Describes the result of a TPU compilation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompilationResultProto {
    /// The error message, if any, returned during compilation.
    #[prost(enumeration="super::error::Code", tag="1")]
    pub status_code: i32,
    #[prost(string, tag="2")]
    pub status_error_message: std::string::String,
    /// HLO proto.
    #[prost(message, repeated, tag="3")]
    pub hlo_protos: ::std::vec::Vec<super::super::xla::HloProto>,
}
/// A mapping between the dynamic shape dimension of an input and the arg that
/// represents the real shape.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PaddingMap {
    /// Input arg index with dynamic shapes.
    #[prost(int32, tag="1")]
    pub arg_index: i32,
    /// The dynamic shape dimension index.
    #[prost(int32, tag="2")]
    pub shape_index: i32,
    /// The arg index that dynamic dimension maps to, which represents the value
    /// of the real shape.
    #[prost(int32, tag="3")]
    pub padding_arg_index: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClippingLimits {
    /// -inf if not set
    #[prost(message, optional, tag="1")]
    pub lower: ::std::option::Option<f32>,
    /// +inf if not set
    #[prost(message, optional, tag="2")]
    pub upper: ::std::option::Option<f32>,
}
/// Dynamic learning rate specification in the TPUEmbeddingConfiguration. The
/// actual learning rates are provided as a scalar input list to the
/// SendTPUEmbeddingGradients Op indexed by their tag specified through the
/// following proto.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DynamicLearningRate {
    /// For tables where learning rates are dynamically computed and communicated
    /// to the TPU embedding program, a tag must be specified for the learning
    /// rate.
    ///
    /// The tag must be a non-negative  integer. The total number of unique tags
    /// must be less than or equal to the number of tables in the TPU embedding
    /// configuration (a table does not specify any tag if it uses a constant
    /// learning rate, and specifies exactly one tag if it uses dynamic learning
    /// rates).
    ///
    /// All tags in the range [0, number_of_unique_tags) must be present in the TPU
    /// embedding configuration, i.e. a tag cannot be skipped if a different tag
    /// numerically greater than it is used in the configuration.
    ///
    /// If multiple tables specify the same tag, they *MUST* have
    /// the same dynamic learning rate, for example, their dynamic learning rate
    /// could be computed by the same TensorFlow sub-graph. The partitioning of the
    /// embedding layer would be more optimal if the number_of_unique_tags is as
    /// *LOW* as possible, i.e., if many tables share the same tag.
    ///
    /// The learning_rate input of the SendTPUEmbeddingGradients op is used to
    /// communicate dynamic learning rates to the TPU embedding program.
    /// The learning_rate input is a list of scalars where the size of the list is
    /// equal to the number of unique tags. The learning rate associated with a
    /// particular tag is specified by populating its corresponding index in the
    /// list of learning_rate scalars.
    #[prost(int32, tag="1")]
    pub tag: i32,
}
/// Source of learning rate to use.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LearningRate {
    #[prost(oneof="learning_rate::LearningRate", tags="1, 2")]
    pub learning_rate: ::std::option::Option<learning_rate::LearningRate>,
}
pub mod learning_rate {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum LearningRate {
        #[prost(float, tag="1")]
        Constant(f32),
        #[prost(message, tag="2")]
        Dynamic(super::DynamicLearningRate),
    }
}
// Each optimizer's parameter proto has a link to its documentation and CPU
// implementation (if available) for user reference.

/// https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L151
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AdagradParameters {
    #[prost(float, tag="1")]
    pub initial_accumulator: f32,
}
/// Algorithm in http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BoundedAdagradParameters {
    /// Whether to use the updated or the old value of the accumulator when
    /// computing the effective learning rate. When update_accumulator_first is set
    /// to True, the updated value of the accumulator is used.
    #[prost(bool, tag="1")]
    pub update_accumulator_first: bool,
    /// The max_var_update value to use. Set value to 0 (default) to disable using
    /// max_var_update to clip the gradient.
    #[prost(float, tag="2")]
    pub max_var_update: f32,
    /// The maximum value of the accumulator. Set max_accumulator to 0 (default)
    /// to disable using max_accumulator to clip the accumulator.
    #[prost(float, tag="3")]
    pub max_accumulator: f32,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L423
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StochasticGradientDescentParameters {
}
/// https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L192
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FtrlParameters {
    #[prost(float, tag="1")]
    pub l1: f32,
    #[prost(float, tag="2")]
    pub l2: f32,
    #[prost(float, tag="3")]
    pub lr_power: f32,
    #[prost(float, tag="4")]
    pub initial_accum: f32,
    #[prost(float, tag="5")]
    pub initial_linear: f32,
}
/// The Adam optimizer does not implement hyper-parameter update; use the dynamic
/// learning rate feature instead, setting the learning rate to:
/// user learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
/// Here, t is the current timestep.
///
/// https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
/// https://github.com/tensorflow/tensorflow/blob/ab51450c817674c8ff08a7ae4f8ac50cdc4bed8b/tensorflow/python/training/adam.py#L54
///
/// Note that the code by default implements the lazy version of Adam
/// (https://www.tensorflow.org/api_docs/python/tf/contrib/opt/LazyAdamOptimizer)
/// unless the use_non_lazy_adam parameter is set, in which case it implements
/// the normal version of Adam that updates all parameters in the embedding
/// table, even for entries that are not used in the current minibatch
/// (https://www.tensorflow.org/api_docs/python/tf/contrib/opt/AdamOptimizer). If
/// use_non_lazy_adam is enabled, gradient accumulation is also required to be
/// enabled in order to get correct results; a warning will be printed otherwise
/// (which may change to an error in the future). If use_sum_inside_sqrt is set,
/// the Adam variable update formula will be changed from m / (sqrt(v) + epsilon)
/// to m / sqrt(v + epsilon**2); this option improves the performance of TPU
/// training and is not expected to harm model quality.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AdamParameters {
    #[prost(float, tag="3")]
    pub beta1: f32,
    #[prost(float, tag="4")]
    pub beta2: f32,
    #[prost(float, tag="5")]
    pub epsilon: f32,
    #[prost(float, tag="6")]
    pub initial_m: f32,
    #[prost(float, tag="7")]
    pub initial_v: f32,
    #[prost(bool, tag="8")]
    pub use_non_lazy_adam: bool,
    #[prost(bool, tag="10")]
    pub use_sum_inside_sqrt: bool,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L271
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MomentumParameters {
    #[prost(float, tag="1")]
    pub momentum: f32,
    #[prost(bool, tag="2")]
    pub use_nesterov: bool,
    #[prost(float, tag="3")]
    pub initial_accum: f32,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L356
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RmsPropParameters {
    #[prost(float, tag="1")]
    pub rho: f32,
    #[prost(float, tag="2")]
    pub momentum: f32,
    #[prost(float, tag="3")]
    pub epsilon: f32,
    #[prost(float, tag="4")]
    pub initial_ms: f32,
    #[prost(float, tag="5")]
    pub initial_mom: f32,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L372
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CenteredRmsPropParameters {
    #[prost(float, tag="1")]
    pub rho: f32,
    #[prost(float, tag="2")]
    pub momentum: f32,
    #[prost(float, tag="3")]
    pub epsilon: f32,
    #[prost(float, tag="4")]
    pub initial_ms: f32,
    #[prost(float, tag="5")]
    pub initial_mom: f32,
    #[prost(float, tag="6")]
    pub initial_mg: f32,
}
/// Variant of algorithm in http://proceedings.mlr.press/v44/shamir15.pdf
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MdlAdagradLightParameters {
    #[prost(float, tag="1")]
    pub l2: f32,
    #[prost(float, tag="2")]
    pub lr_power: f32,
    #[prost(float, tag="3")]
    pub min_servable_mdl_benefit: f32,
    #[prost(float, tag="4")]
    pub mdl_mix_in_margin: f32,
    #[prost(float, tag="5")]
    pub mdl_benefit_rampup_coeff: f32,
    #[prost(float, tag="6")]
    pub mdl_min_weight: f32,
    #[prost(float, tag="7")]
    pub benefit_revisit_scale: f32,
    #[prost(float, tag="8")]
    pub max_event_benefit: f32,
    #[prost(float, tag="9")]
    pub max_total_benefit: f32,
    #[prost(float, tag="10")]
    pub mdl_hard_limit: f32,
    #[prost(bool, tag="11")]
    pub hard_limit_min_benefit: bool,
    #[prost(bool, tag="12")]
    pub mdl_regularize: bool,
    #[prost(float, tag="13")]
    pub initial_accumulator: f32,
    #[prost(float, tag="14")]
    pub initial_weight: f32,
    #[prost(float, tag="15")]
    pub initial_benefit: f32,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L68
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AdadeltaParameters {
    #[prost(float, tag="1")]
    pub rho: f32,
    #[prost(float, tag="2")]
    pub epsilon: f32,
    #[prost(float, tag="3")]
    pub initial_accumulator: f32,
    #[prost(float, tag="4")]
    pub initial_update: f32,
}
/// https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer
/// https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/core/kernels/training_ops.cc#L164
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProximalAdagradParameters {
    #[prost(float, tag="1")]
    pub l1: f32,
    #[prost(float, tag="2")]
    pub l2: f32,
    #[prost(float, tag="3")]
    pub initial_accumulator: f32,
}
/// The online Yogi optimizer does not implement hyper-parameter update; use the
/// dynamic learning rate feature instead, setting the learning rate to:
/// user learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
/// Here, t is the current timestep.
///
/// https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
/// plus some extensions based on FTRL.
///
/// Note that the code by default implements the lazy version of online Yogi.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OnlineYogiParameters {
    /// The L1 regularization parameter (used analogously to the one in FTRL).
    #[prost(float, tag="1")]
    pub l1: f32,
    /// The L2 regularization parameter (used analogously to the one in FTRL).
    #[prost(float, tag="2")]
    pub l2: f32,
    /// \beta_2 from Algorithm 2 in the paper.
    #[prost(float, tag="3")]
    pub beta2: f32,
    /// Initial value of V variable in paper.
    #[prost(float, tag="4")]
    pub initial_v: f32,
    /// Initial value of linear variable in FTRL.
    #[prost(float, tag="5")]
    pub initial_linear: f32,
    /// Activation to use to replace sign function in v_t update in Algorithm 2 of
    /// paper.
    #[prost(oneof="online_yogi_parameters::Activation", tags="6, 7")]
    pub activation: ::std::option::Option<online_yogi_parameters::Activation>,
}
pub mod online_yogi_parameters {
    /// x -> copysign(1, x) (i.e., return 1 for an input of +0 rather than 0).
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SignActivation {
    }
    /// x -> tanh(x * 10)
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TanhActivation {
    }
    /// Activation to use to replace sign function in v_t update in Algorithm 2 of
    /// paper.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Activation {
        #[prost(message, tag="6")]
        Sign(SignActivation),
        #[prost(message, tag="7")]
        Tanh(TanhActivation),
    }
}
/// Status of using gradient accumulation (doing two passes over the input
/// gradients: one to accumulate them into a temporary array and another to apply
/// them using the actual optimization algorithm). The extra message is to wrap
/// the enum for scoping.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GradientAccumulationStatus {
}
pub mod gradient_accumulation_status {
    /// if UNSPECIFIED (default), gradient accumulation is ENABLED.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Status {
        Unspecified = 0,
        Enabled = 1,
        Disabled = 2,
    }
}
/// Configuration proto for hot ID optimization. This is an experimental feature
/// that is currently disabled (by default).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HotIdReplicationConfiguration {
    #[prost(enumeration="hot_id_replication_configuration::Status", tag="1")]
    pub status: i32,
}
pub mod hot_id_replication_configuration {
    /// Whether to enable or disable hot ID optimization.
    /// If UNSPECIFIED (default), hot ID optimization is DISABLED.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Status {
        Unspecified = 0,
        Enabled = 1,
        Disabled = 2,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OptimizationParameters {
    /// Learning rate used for updating the embedding layer parameters.
    #[prost(message, optional, tag="13")]
    pub learning_rate: ::std::option::Option<LearningRate>,
    /// Limits to which to clip the weight values after the backward pass; not
    /// present means no limits are applied.
    #[prost(message, optional, tag="2")]
    pub clipping_limits: ::std::option::Option<ClippingLimits>,
    /// Limits to which to clip the backward pass gradient before using it for
    /// updates; not present means no limits are applied.
    #[prost(message, optional, tag="7")]
    pub gradient_clipping_limits: ::std::option::Option<ClippingLimits>,
    /// Amount of weight decay to apply; see weight_decay_optimizers.py for
    /// details. Almost all optimizers are supported with this option (MDL Adagrad
    /// Light does not work, and SGD does not behave as expected if it is enabled).
    /// Although there is no check, users who want weight decay will probably also
    /// want to enable gradient accumulation as well so that the decay will happen
    /// once per minibatch.
    #[prost(float, tag="16")]
    pub weight_decay_factor: f32,
    /// Status of using gradient accumulation (doing two passes over the input
    /// gradients: one to accumulate them into a temporary array and another to
    /// apply them using the actual optimization algorithm).
    #[prost(enumeration="gradient_accumulation_status::Status", tag="17")]
    pub gradient_accumulation_status: i32,
    /// Configuration proto for hot ID replication. This is an experimental
    /// feature that is currently disabled (by default).
    #[prost(message, optional, tag="18")]
    pub hot_id_replication_configuration: ::std::option::Option<HotIdReplicationConfiguration>,
    /// Optimization algorithm parameters; which field is selected determines which
    /// algorithm to use.
    #[prost(oneof="optimization_parameters::Parameters", tags="3, 19, 4, 5, 6, 8, 9, 10, 11, 12, 14, 20")]
    pub parameters: ::std::option::Option<optimization_parameters::Parameters>,
}
pub mod optimization_parameters {
    /// Optimization algorithm parameters; which field is selected determines which
    /// algorithm to use.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Parameters {
        #[prost(message, tag="3")]
        Adagrad(super::AdagradParameters),
        #[prost(message, tag="19")]
        BoundedAdagrad(super::BoundedAdagradParameters),
        #[prost(message, tag="4")]
        StochasticGradientDescent(super::StochasticGradientDescentParameters),
        #[prost(message, tag="5")]
        Ftrl(super::FtrlParameters),
        #[prost(message, tag="6")]
        Adam(super::AdamParameters),
        #[prost(message, tag="8")]
        Momentum(super::MomentumParameters),
        #[prost(message, tag="9")]
        RmsProp(super::RmsPropParameters),
        #[prost(message, tag="10")]
        CenteredRmsProp(super::CenteredRmsPropParameters),
        #[prost(message, tag="11")]
        MdlAdagradLight(super::MdlAdagradLightParameters),
        #[prost(message, tag="12")]
        Adadelta(super::AdadeltaParameters),
        #[prost(message, tag="14")]
        ProximalAdagrad(super::ProximalAdagradParameters),
        #[prost(message, tag="20")]
        OnlineYogi(super::OnlineYogiParameters),
    }
}
/// Specification of an optimization algorithm's state variables (both the main
/// value vector and any extra accumulators, etc.). This proto is only used
/// internally by the TPU software and is not exposed directly to the TF model.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StateVariableSpecification {
    /// Parameter name for the state variable.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// Usage type of this state variable.
    #[prost(oneof="state_variable_specification::Usage", tags="2, 3")]
    pub usage: ::std::option::Option<state_variable_specification::Usage>,
}
pub mod state_variable_specification {
    /// A normal state variable that should be saved and restored in checkpoints
    /// and used as an input or output to non-debug TensorFlow ops.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct UserDefined {
        /// For padding embedding rows, this field specifies the initial value to be
        /// used. Separate initial values need to be specified for the embeddings and
        /// any extra accumulators. The initial values should be specified so as to
        /// maintain two invariants during model training:
        /// (1) The embedding vector multiplied by zero returns a vector containing
        ///     all zeros. To maintain this invariant, the embedding values should
        ///     never be NaNs or +-infinity.
        /// (2) Repeatedly applying the optimizer using a gradient vector of all
        ///     zeros does not cause the embeddings or slot variables to become NaNs
        ///     or +-infinity.
        /// The padding row is looked up when no embedding IDs are present for a
        /// feature. The semantics of embedding lookup dictate that the output must
        /// be zero under this scenario.
        #[prost(double, tag="1")]
        pub padding_initial_value: f64,
    }
    /// A state variable that should be filled with a constant and normally hidden
    /// from users (used for intermediate gradients being accumulated, for
    /// example).
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FillWithConstant {
        #[prost(double, tag="1")]
        pub initial_value: f64,
    }
    /// Usage type of this state variable.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Usage {
        #[prost(message, tag="2")]
        UserDefined(UserDefined),
        #[prost(message, tag="3")]
        FillWithConstant(FillWithConstant),
    }
}
// In the comments here, "layout" refers to the top-level EmbeddingOutputLayout
// proto contained in the TPUEmbeddingConfiguration.

// The embedding output consists of a list of tensors, each specified by an
// EmbeddingOutputTensor proto within the EmbeddingOutputLayout (the "output"
// field). Each table and feature lookup is then placed into some number of
// particular positions within some output tensor (identified by "tensor_index"
// within OutputLocation). The tree of table lookups, feature lookups, and
// output locations is specified by the
// "table(table_id).feature(feature_id).output_location" repeated fields within
// EmbeddingOutputLayout.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TpuEmbeddingOutputLayout {
    /// Output locations for each feature of each table.
    #[prost(message, repeated, tag="1")]
    pub table: ::std::vec::Vec<tpu_embedding_output_layout::TableDescriptor>,
    /// Shape and layout information for each tensor.
    #[prost(message, repeated, tag="2")]
    pub output: ::std::vec::Vec<tpu_embedding_output_layout::EmbeddingOutputTensor>,
}
pub mod tpu_embedding_output_layout {
    /// Location of one copy of the feature's data.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct OutputLocation {
        /// Which output tensor this copy of the feature will go into. Must be
        /// between 0 and layout.output_size().
        #[prost(int32, tag="1")]
        pub tensor_index: i32,
        /// Offset in dimension 0 for this feature copy. Must be between 0 and
        /// layout.output(tensor_index).dim0_size_per_sample().
        #[prost(int32, tag="2")]
        pub dim0_offset: i32,
        /// Offset in dimension 1 for this feature copy. Must be between 0 and
        /// layout.output(tensor_index).dim1_size() - table width; repeated or
        /// partially/fully overlapping values are allowed and results in the same
        /// range will be summed (with the gradients replicated in the backward
        /// pass).
        #[prost(int32, tag="3")]
        pub dim1_offset: i32,
    }
    /// Description of the output placement for one feature.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FeatureDescriptor {
        /// Typically, only one copy of each feature is used, but multiple are
        /// allowed and the same data will be copied to all of them (with the
        /// gradients summed in the backward pass).
        #[prost(message, repeated, tag="1")]
        pub output_location: ::std::vec::Vec<OutputLocation>,
    }
    /// Description of the output placement for features of one table.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TableDescriptor {
        /// Output locations for each feature loaded from this table.
        #[prost(message, repeated, tag="1")]
        pub feature: ::std::vec::Vec<FeatureDescriptor>,
    }
    // Data layout and shape computation information for a single output tensor.
    // Any unused locations in the tensor will be filled with zeros, and
    // corresponding gradients will be ignored.

    /// Size and layout information for 2-D tensors.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TwoDOutputTensor {
        /// Multiplier for output dimension 0 size; used to match legacy format that
        /// stacks features within a sample in dimension 0.
        #[prost(int32, tag="2")]
        pub dim0_size_per_sample: i32,
        /// The size (in dimension 1) of this output tensor.
        #[prost(int32, tag="1")]
        pub dim1_size: i32,
    }
    /// Format information for a single output tensor.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct EmbeddingOutputTensor {
        #[prost(oneof="embedding_output_tensor::OutputFormat", tags="4")]
        pub output_format: ::std::option::Option<embedding_output_tensor::OutputFormat>,
    }
    pub mod embedding_output_tensor {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum OutputFormat {
            #[prost(message, tag="4")]
            TwoD(super::TwoDOutputTensor),
        }
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TpuEmbeddingConfiguration {
    #[prost(message, repeated, tag="1")]
    pub table_descriptor: ::std::vec::Vec<tpu_embedding_configuration::TableDescriptor>,
    #[prost(enumeration="tpu_embedding_configuration::Mode", tag="2")]
    pub mode: i32,
    /// Number of samples in each batch of embedding layer activations sent to
    /// the TensorCore.
    #[prost(int32, tag="3")]
    pub batch_size_per_tensor_core: i32,
    /// Number of TPU hosts used for inference/training.
    #[prost(int32, tag="4")]
    pub num_hosts: i32,
    /// Number of TensorCore used for inference/training.
    #[prost(int32, tag="5")]
    pub num_tensor_cores: i32,
    #[prost(enumeration="tpu_embedding_configuration::ShardingStrategy", tag="6")]
    pub sharding_strategy: i32,
    /// This parameter determines if the execution of the sparse core will be
    /// pipelined with that of the TensorCore. This parameter only affects results
    /// when mode=TRAINING. If mode=INFERENCE or BACKWARD_PASS_ONLY, this parameter
    /// does not affect execution and hence, is a don't care value.
    ///
    /// false: The execution of the sparse core is not pipelined with that of the
    /// TensorCore. The forward pass of every step on the sparse core is executed
    /// only after the backward pass of the previous step is complete. And the
    /// backward pass on the sparse core is executed only after the embedding
    /// gradients have been computed on the TensorCore on every step. This ensures
    /// that the activations on every step observe the gradient updates from the
    /// previous step on both the sparse core and the TensorCore.
    ///
    /// true: The execution of the sparse core is pipelined with that of the
    /// TensorCore. The forward pass of every step on the sparse core can be
    /// executed after the forward pass of the previous step is complete without
    /// waiting for the backward pass. This improves the utilization of the sparse
    /// core allowing it to process step N+1 while the embedding gradients for step
    /// N are computed on the TensorCore. The backward pass of every step on the
    /// sparse core is executed directly after the forward pass for the next step
    /// is complete. The drawback is that embedding activations for step N+1 do not
    /// observe the embedding gradient updates from step N. This could affect model
    /// quality if step N and N+1 involve the same set of embedding IDs. However,
    /// since the embedding updates are sparse, this is generally not considered a
    /// problem.
    #[prost(bool, tag="7")]
    pub pipeline_execution_with_tensor_core: bool,
    /// Extended output layout information; if not provided, a compatibility mode
    /// will use defaults that match the old layout. Providing a value for this
    /// field is EXPERIMENTAL and most ways of filling it will probably break. Do
    /// not set it unless you know what you are doing.
    #[prost(message, optional, tag="8")]
    pub output_layout: ::std::option::Option<TpuEmbeddingOutputLayout>,
}
pub mod tpu_embedding_configuration {
    /// Description of the various embedding tables.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TableDescriptor {
        /// Name of the table.
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// Size of the vocabulary (i.e., number of rows) in the table.
        #[prost(int32, tag="2")]
        pub vocabulary_size: i32,
        /// The embedding dimension (i.e., the width of the embedding table).
        #[prost(int32, tag="3")]
        pub dimension: i32,
        /// Number of features mapped to this table.
        #[prost(int32, tag="4")]
        pub num_features: i32,
        /// Details of the learning algorithm used to update the embedding
        /// parameters.
        #[prost(message, optional, tag="5")]
        pub optimization_parameters: ::std::option::Option<super::OptimizationParameters>,
    }
    /// Mode. Should the embedding layer program be run for inference (just forward
    /// pass), training (both forward and backward pass) or just the backward_pass.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Mode {
        Unspecified = 0,
        Inference = 1,
        Training = 2,
        BackwardPassOnly = 3,
    }
    /// Sharding strategy of the embedding tables among the hosts.
    /// If the sharding_strategy is "mod", each id is assigned to host
    /// "id % num_hosts". For instance, 13 ids are split across 5 hosts as:
    /// [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]].
    /// If the sharding_strategy is "div", ids are assigned to hosts in a
    /// contiguous manner. In this case, 13 ids are split across 5 hosts as:
    /// [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]].
    /// In both the strategies, if the id space does not evenly divide the number
    /// of hosts, each of the first "table_descriptor.vocabulary_size % num_hosts"
    /// hosts will be assigned one more id.
    /// This partitioning strategy exactly follows that in the embedding_lookup
    /// TensorFlow function at tensorflow/python/ops/embedding_ops.py.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum ShardingStrategy {
        DivDefault = 0,
        Mod = 1,
    }
}
/// Describes the geometry of a TPU mesh.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TopologyProto {
    /// The dimensions of the TPU topology, in cores. Typically, this is a 3D
    /// topology [x, y, core], where the major dimensions correspond to TPU chips,
    /// and the minor dimension describes the number of cores on a multicore chip.
    #[prost(int32, repeated, tag="1")]
    pub mesh_shape: ::std::vec::Vec<i32>,
    /// Number of TensorFlow tasks in the cluster.
    #[prost(int32, tag="2")]
    pub num_tasks: i32,
    /// Number of TPU devices per task.
    #[prost(int32, tag="3")]
    pub num_tpu_devices_per_task: i32,
    /// A flattened rank 3 int32 array with shape
    /// [num_tasks, num_tpu_devices_per_task, len(mesh_shape)].
    /// `tasks` is the number of tasks in the TPU cluster, `devices` is the number
    /// of TPU devices per task, and the minor dimension corresponds to a position
    /// in the TPU mesh topology. Each entry [task, device, axis] gives the
    /// `axis`-th coordinate in the topology of a task/device pair.
    #[prost(int32, repeated, tag="4")]
    pub device_coordinates: ::std::vec::Vec<i32>,
}
