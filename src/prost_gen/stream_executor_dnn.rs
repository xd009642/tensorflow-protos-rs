/// Generic tensor representation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorDescriptorProto {
    #[prost(int64, repeated, tag="1")]
    pub dimensions: ::std::vec::Vec<i64>,
    #[prost(enumeration="DataType", tag="2")]
    pub data_type: i32,
    #[prost(oneof="tensor_descriptor_proto::LayoutOneof", tags="3, 4")]
    pub layout_oneof: ::std::option::Option<tensor_descriptor_proto::LayoutOneof>,
}
pub mod tensor_descriptor_proto {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum LayoutOneof {
        #[prost(enumeration="super::DataLayout", tag="3")]
        DataLayout(i32),
        #[prost(enumeration="super::FilterLayout", tag="4")]
        FilterLayout(i32),
    }
}
/// Generic algorithm representation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AlgorithmProto {
    #[prost(int64, tag="1")]
    pub algo_id: i64,
    #[prost(enumeration="algorithm_proto::MathType", tag="2")]
    pub math_type: i32,
}
pub mod algorithm_proto {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum MathType {
        DefaultMath = 0,
        /// The GPU may operate 4x4 matrix FMA.
        /// See cuDNN's documentation for CUDNN_TENSOR_OP_MATH.
        TensorOpMath = 1,
    }
}
/// Convolution-specific parameters.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConvolutionDescriptorProto {
    #[prost(int64, repeated, tag="1")]
    pub paddings: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="2")]
    pub strides: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="3")]
    pub dilations: ::std::vec::Vec<i64>,
    /// The "accumulator" type. For example, use F32 as an accumulator for F16
    /// convolutions.
    /// See cuDNN's cudnnConvolutionMode_t.
    #[prost(enumeration="DataType", tag="4")]
    pub compute_mode: i32,
    /// See cuDNN's group count.
    #[prost(int32, tag="5")]
    pub group_count: i32,
    #[prost(enumeration="ConvolutionMode", tag="6")]
    pub convolution_mode: i32,
    /// Tensorflow node name, same as in NodeDef, for debugging purposes.
    #[prost(string, tag="7")]
    pub name: std::string::String,
}
/// Specifies the data type used by an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum DataType {
    KFloat = 0,
    KDouble = 1,
    KHalf = 2,
    KInt8 = 3,
    KInt32 = 4,
}
/// Describes how a convolution input or output layer's data is formatted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum DataLayout {
    /// Naming convention:
    /// Y <-> row or height
    /// X <-> column or width
    /// Batch <-> batch, or N
    /// Depth <-> feature, or channel
    /// TODO(timshen): turn them into cuDNN names, e.g. kNCHW.
    KYxDepthBatch = 0,
    KYxBatchDepth = 1,
    /// cuDNN's NHWC layout
    KBatchYxDepth = 2,
    /// cuDNN's NCHW layout
    KBatchDepthYx = 3,
    /// cuDNN's NCHW_VECT_C layout
    KBatchDepthYx4 = 4,
}
/// Describes how a convolution filter is laid out in the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum FilterLayout {
    /// Naming convention:
    /// Y <-> row or height
    /// X <-> column or width
    /// Output <-> output feature, or N
    /// Input <-> input feature, or N
    /// TODO(timshen): turn them into cuDNN names, e.g. kNCHW.
    ///
    /// cuDNN's NCHW layout
    KOutputInputYx = 0,
    /// cuDNN's NHWC layout
    KOutputYxInput = 1,
    /// cuDNN's NCHW_VECT_C layout
    KOutputInputYx4 = 2,
    KInputYxOutput = 3,
    KYxInputOutput = 4,
}
/// Describes a kind of non-linearity (threshold-like mathematical function).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ActivationMode {
    KNone = 0,
    KSigmoid = 1,
    /// Rectified linear activation: f(x) = x < 0 ? 0 : x
    KRelu = 2,
    /// Rectified linear activation; where upper maximum is 6.0.
    KRelu6 = 3,
    /// Rectified linear activation; where upper maximum specified by
    /// BatchDescriptor::value_max().
    KReluX = 4,
    KTanh = 5,
    /// Like ReluX; but passes all values in the range [-X,X].
    KBandPass = 6,
}
/// Describe the math definition for the conv op. The popular behavior is
/// actually called cross-correlation in math, despite the operation is often
/// referred as convolution. See cuDNN cudnnConvolutionMode_t.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ConvolutionMode {
    CrossCorrelation = 0,
    Convolution = 1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ConvolutionKind {
    Invalid = 0,
    Forward = 1,
    BackwardFilter = 2,
    BackwardData = 3,
    ForwardBiasActivation = 4,
}
