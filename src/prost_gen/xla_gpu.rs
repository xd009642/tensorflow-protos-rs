#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConvInstructionLog {
    #[prost(message, optional, tag="1")]
    pub instruction: ::std::option::Option<super::HloInstructionProto>,
    #[prost(message, repeated, tag="2")]
    pub operand_shapes: ::std::vec::Vec<super::ShapeProto>,
    #[prost(uint64, tag="3")]
    pub result_address: u64,
    #[prost(uint64, repeated, tag="4")]
    pub operand_addresses: ::std::vec::Vec<u64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlacklistedAlgorithm {
    #[prost(int64, tag="1")]
    pub id: i64,
    #[prost(bool, tag="2")]
    pub tensor_ops: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AlgorithmBlacklistEntry {
    #[prost(string, tag="1")]
    pub hlo: std::string::String,
    #[prost(message, optional, tag="2")]
    pub cc: ::std::option::Option<super::super::tensorflow::ComputeCapability>,
    #[prost(message, optional, tag="3")]
    pub cudnn_version: ::std::option::Option<super::super::tensorflow::CudnnVersion>,
    #[prost(string, tag="5")]
    pub blas_version: std::string::String,
    #[prost(message, repeated, tag="4")]
    pub algos: ::std::vec::Vec<BlacklistedAlgorithm>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AlgorithmBlacklist {
    #[prost(message, repeated, tag="1")]
    pub entries: ::std::vec::Vec<AlgorithmBlacklistEntry>,
}
// Backend configs for XLA:GPU.
//
// These are metadata that the GPU backend attaches to HloInstrucitons and later
// uses during e.g. codegen.
//
// Remember that proto3 doesn't give clients a way to tell the difference
// between a field not being present and a field having the default value.
// Choose your defaults carefully.
//
// No guarantee is made about the stability of these protos.
//
// See HloInstruction::backend_config() for more info.

/// Backend config for a convolution that runs through cudnn.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CudnnConvBackendConfig {
    /// Opaque algorithm number of cudnn algorithm chosen for this conv.
    #[prost(int64, tag="1")]
    pub algorithm: i64,
    /// Whether we may use tensor cores when running this conv.  Even if this is
    /// true, cudnn may choose not to use tensor cores, e.g. because the GPU or
    /// selected algorithm doesn't support it.
    #[prost(bool, tag="2")]
    pub tensor_ops_enabled: bool,
    /// The scaling factor multiplied with the convolution result.
    #[prost(double, tag="4")]
    pub conv_result_scale: f64,
    // Below are the fields related to cuDNN's fused convolution. Refer to
    // CudnnConvParams for their meanings.

    /// The requested activation (e.g. relu) after the convolution. It is with type
    /// stream_executor::dnn::ActivationMode.
    #[prost(int64, tag="3")]
    pub activation_mode: i64,
    /// The scaling factor multiplied with the side input. If no side input buffer
    /// is provided, this field must be 0.
    #[prost(double, tag="5")]
    pub side_input_scale: f64,
}
/// Backend config for the GEMM operation running through cuBLAS.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GemmBackendConfig {
    #[prost(double, tag="2")]
    pub alpha_real: f64,
    #[prost(double, tag="9")]
    pub alpha_imag: f64,
    #[prost(double, tag="3")]
    pub beta: f64,
    #[prost(message, optional, tag="7")]
    pub dot_dimension_numbers: ::std::option::Option<super::DotDimensionNumbers>,
    #[prost(int64, tag="8")]
    pub batch_size: i64,
    /// Opaque optional algorithm number. No chosen number indicates that a
    /// different cuBLAS API will be used, which does not allow for choosing an
    /// algorithm.
    #[prost(oneof="gemm_backend_config::Algorithm", tags="1")]
    pub algorithm: ::std::option::Option<gemm_backend_config::Algorithm>,
}
pub mod gemm_backend_config {
    /// Opaque optional algorithm number. No chosen number indicates that a
    /// different cuBLAS API will be used, which does not allow for choosing an
    /// algorithm.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Algorithm {
        #[prost(int64, tag="1")]
        SelectedAlgorithm(i64),
    }
}
