/// Dimensions of a tensor.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    /// Dimensions of the tensor, such as {"input", 30}, {"output", 40}
    /// for a 30 x 40 2D tensor.  If an entry has size -1, this
    /// corresponds to a dimension of unknown size. The names are
    /// optional.
    ///
    /// The order of entries in "dim" matters: It indicates the layout of the
    /// values in the tensor in-memory representation.
    ///
    /// The first entry in "dim" is the outermost dimension used to layout the
    /// values, the last entry is the innermost dimension.  This matches the
    /// in-memory layout of RowMajor Eigen tensors.
    ///
    /// If "dim.size()" > 0, "unknown_rank" must be false.
    #[prost(message, repeated, tag="2")]
    pub dim: ::std::vec::Vec<tensor_shape_proto::Dim>,
    /// If true, the number of dimensions in the shape is unknown.
    ///
    /// If true, "dim.size()" must be 0.
    #[prost(bool, tag="3")]
    pub unknown_rank: bool,
}
pub mod tensor_shape_proto {
    /// One dimension of the tensor.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Dim {
        /// Size of the tensor in that dimension.
        /// This value must be >= -1, but values of -1 are reserved for "unknown"
        /// shapes (values of -1 mean "unknown" dimension).  Certain wrappers
        /// that work with TensorShapeProto may fail at runtime when deserializing
        /// a TensorShapeProto containing a dim value of -1.
        #[prost(int64, tag="1")]
        pub size: i64,
        /// Optional name of the tensor dimension.
        #[prost(string, tag="2")]
        pub name: std::string::String,
    }
}
/// (== suppress_warning documentation-presence ==)
/// LINT.IfChange
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum DataType {
    /// Not a legal value for DataType.  Used to indicate a DataType field
    /// has not been set.
    DtInvalid = 0,
    /// Data types that all computation devices are expected to be
    /// capable to support.
    DtFloat = 1,
    DtDouble = 2,
    DtInt32 = 3,
    DtUint8 = 4,
    DtInt16 = 5,
    DtInt8 = 6,
    DtString = 7,
    /// Single-precision complex
    DtComplex64 = 8,
    DtInt64 = 9,
    DtBool = 10,
    /// Quantized int8
    DtQint8 = 11,
    /// Quantized uint8
    DtQuint8 = 12,
    /// Quantized int32
    DtQint32 = 13,
    /// Float32 truncated to 16 bits.  Only for cast ops.
    DtBfloat16 = 14,
    /// Quantized int16
    DtQint16 = 15,
    /// Quantized uint16
    DtQuint16 = 16,
    DtUint16 = 17,
    /// Double-precision complex
    DtComplex128 = 18,
    DtHalf = 19,
    DtResource = 20,
    /// Arbitrary C++ data types
    DtVariant = 21,
    DtUint32 = 22,
    DtUint64 = 23,
    /// Do not use!  These are only for parameters.  Every enum above
    /// should have a corresponding value below (verified by types_test).
    DtFloatRef = 101,
    DtDoubleRef = 102,
    DtInt32Ref = 103,
    DtUint8Ref = 104,
    DtInt16Ref = 105,
    DtInt8Ref = 106,
    DtStringRef = 107,
    DtComplex64Ref = 108,
    DtInt64Ref = 109,
    DtBoolRef = 110,
    DtQint8Ref = 111,
    DtQuint8Ref = 112,
    DtQint32Ref = 113,
    DtBfloat16Ref = 114,
    DtQint16Ref = 115,
    DtQuint16Ref = 116,
    DtUint16Ref = 117,
    DtComplex128Ref = 118,
    DtHalfRef = 119,
    DtResourceRef = 120,
    DtVariantRef = 121,
    DtUint32Ref = 122,
    DtUint64Ref = 123,
}
/// Protocol buffer representing a handle to a tensorflow resource. Handles are
/// not valid across executions, but can be serialized back and forth from within
/// a single run.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResourceHandleProto {
    /// Unique name for the device containing the resource.
    #[prost(string, tag="1")]
    pub device: std::string::String,
    /// Container in which this resource is placed.
    #[prost(string, tag="2")]
    pub container: std::string::String,
    /// Unique name of this resource.
    #[prost(string, tag="3")]
    pub name: std::string::String,
    /// Hash code for the type of the resource. Is only valid in the same device
    /// and in the same execution.
    #[prost(uint64, tag="4")]
    pub hash_code: u64,
    /// For debug-only, the name of the type pointed to by this handle, if
    /// available.
    #[prost(string, tag="5")]
    pub maybe_type_name: std::string::String,
    /// Data types and shapes for the underlying resource.
    #[prost(message, repeated, tag="6")]
    pub dtypes_and_shapes: ::std::vec::Vec<resource_handle_proto::DtypeAndShape>,
}
pub mod resource_handle_proto {
    /// Protocol buffer representing a pair of (data type, tensor shape).
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DtypeAndShape {
        #[prost(enumeration="super::DataType", tag="1")]
        pub dtype: i32,
        #[prost(message, optional, tag="2")]
        pub shape: ::std::option::Option<super::TensorShapeProto>,
    }
}
/// Protocol buffer representing a tensor.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    #[prost(enumeration="DataType", tag="1")]
    pub dtype: i32,
    /// Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
    #[prost(message, optional, tag="2")]
    pub tensor_shape: ::std::option::Option<TensorShapeProto>,
    // Only one of the representations below is set, one of "tensor_contents" and
    // the "xxx_val" attributes.  We are not using oneof because as oneofs cannot
    // contain repeated fields it would require another extra set of messages.

    /// Version number.
    ///
    /// In version 0, if the "repeated xxx" representations contain only one
    /// element, that element is repeated to fill the shape.  This makes it easy
    /// to represent a constant Tensor with a single value.
    #[prost(int32, tag="3")]
    pub version_number: i32,
    /// Serialized raw tensor content from either Tensor::AsProtoTensorContent or
    /// memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
    /// can be used for all tensor types. The purpose of this representation is to
    /// reduce serialization overhead during RPC call by avoiding serialization of
    /// many repeated small items.
    #[prost(bytes, tag="4")]
    pub tensor_content: std::vec::Vec<u8>,
    // Type specific representations that make it easy to create tensor protos in
    // all languages.  Only the representation corresponding to "dtype" can
    // be set.  The values hold the flattened representation of the tensor in
    // row major order.

    /// DT_HALF, DT_BFLOAT16. Note that since protobuf has no int16 type, we'll
    /// have some pointless zero padding for each value here.
    #[prost(int32, repeated, tag="13")]
    pub half_val: ::std::vec::Vec<i32>,
    /// DT_FLOAT.
    #[prost(float, repeated, tag="5")]
    pub float_val: ::std::vec::Vec<f32>,
    /// DT_DOUBLE.
    #[prost(double, repeated, tag="6")]
    pub double_val: ::std::vec::Vec<f64>,
    /// DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
    #[prost(int32, repeated, tag="7")]
    pub int_val: ::std::vec::Vec<i32>,
    /// DT_STRING
    #[prost(bytes, repeated, tag="8")]
    pub string_val: ::std::vec::Vec<std::vec::Vec<u8>>,
    /// DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
    /// and imaginary parts of i-th single precision complex.
    #[prost(float, repeated, tag="9")]
    pub scomplex_val: ::std::vec::Vec<f32>,
    /// DT_INT64
    #[prost(int64, repeated, tag="10")]
    pub int64_val: ::std::vec::Vec<i64>,
    /// DT_BOOL
    #[prost(bool, repeated, tag="11")]
    pub bool_val: ::std::vec::Vec<bool>,
    /// DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
    /// and imaginary parts of i-th double precision complex.
    #[prost(double, repeated, tag="12")]
    pub dcomplex_val: ::std::vec::Vec<f64>,
    /// DT_RESOURCE
    #[prost(message, repeated, tag="14")]
    pub resource_handle_val: ::std::vec::Vec<ResourceHandleProto>,
    /// DT_VARIANT
    #[prost(message, repeated, tag="15")]
    pub variant_val: ::std::vec::Vec<VariantTensorDataProto>,
    /// DT_UINT32
    #[prost(uint32, repeated, tag="16")]
    pub uint32_val: ::std::vec::Vec<u32>,
    /// DT_UINT64
    #[prost(uint64, repeated, tag="17")]
    pub uint64_val: ::std::vec::Vec<u64>,
}
/// Protocol buffer representing the serialization format of DT_VARIANT tensors.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VariantTensorDataProto {
    /// Name of the type of objects being serialized.
    #[prost(string, tag="1")]
    pub type_name: std::string::String,
    /// Portions of the object that are not Tensors.
    #[prost(bytes, tag="2")]
    pub metadata: std::vec::Vec<u8>,
    /// Tensors contained within objects being serialized.
    #[prost(message, repeated, tag="3")]
    pub tensors: ::std::vec::Vec<TensorProto>,
}
/// Protocol buffer representing the value for an attr used to configure an Op.
/// Comment indicates the corresponding attr type.  Only the field matching the
/// attr type may be filled.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttrValue {
    #[prost(oneof="attr_value::Value", tags="2, 3, 4, 5, 6, 7, 8, 1, 10, 9")]
    pub value: ::std::option::Option<attr_value::Value>,
}
pub mod attr_value {
    /// LINT.IfChange
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ListValue {
        /// "list(string)"
        #[prost(bytes, repeated, tag="2")]
        pub s: ::std::vec::Vec<std::vec::Vec<u8>>,
        /// "list(int)"
        #[prost(int64, repeated, tag="3")]
        pub i: ::std::vec::Vec<i64>,
        /// "list(float)"
        #[prost(float, repeated, tag="4")]
        pub f: ::std::vec::Vec<f32>,
        /// "list(bool)"
        #[prost(bool, repeated, tag="5")]
        pub b: ::std::vec::Vec<bool>,
        /// "list(type)"
        #[prost(enumeration="super::DataType", repeated, tag="6")]
        pub r#type: ::std::vec::Vec<i32>,
        /// "list(shape)"
        #[prost(message, repeated, tag="7")]
        pub shape: ::std::vec::Vec<super::TensorShapeProto>,
        /// "list(tensor)"
        #[prost(message, repeated, tag="8")]
        pub tensor: ::std::vec::Vec<super::TensorProto>,
        /// "list(attr)"
        #[prost(message, repeated, tag="9")]
        pub func: ::std::vec::Vec<super::NameAttrList>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        /// "string"
        #[prost(bytes, tag="2")]
        S(std::vec::Vec<u8>),
        /// "int"
        #[prost(int64, tag="3")]
        I(i64),
        /// "float"
        #[prost(float, tag="4")]
        F(f32),
        /// "bool"
        #[prost(bool, tag="5")]
        B(bool),
        /// "type"
        #[prost(enumeration="super::DataType", tag="6")]
        Type(i32),
        /// "shape"
        #[prost(message, tag="7")]
        Shape(super::TensorShapeProto),
        /// "tensor"
        #[prost(message, tag="8")]
        Tensor(super::TensorProto),
        /// any "list(...)"
        #[prost(message, tag="1")]
        List(ListValue),
        /// "func" represents a function. func.name is a function's name or
        /// a primitive op's name. func.attr.first is the name of an attr
        /// defined for that function. func.attr.second is the value for
        /// that attr in the instantiation.
        #[prost(message, tag="10")]
        Func(super::NameAttrList),
        /// This is a placeholder only used in nodes defined inside a
        /// function.  It indicates the attr value will be supplied when
        /// the function is instantiated.  For example, let us suppose a
        /// node "N" in function "FN". "N" has an attr "A" with value
        /// placeholder = "foo". When FN is instantiated with attr "foo"
        /// set to "bar", the instantiated node N's attr A will have been
        /// given the value "bar".
        #[prost(string, tag="9")]
        Placeholder(std::string::String),
    }
}
/// A list of attr names and their values. The whole list is attached
/// with a string name.  E.g., MatMul[T=float].
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NameAttrList {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(map="string, message", tag="2")]
    pub attr: ::std::collections::HashMap<std::string::String, AttrValue>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AllocationDescription {
    /// Total number of bytes requested
    #[prost(int64, tag="1")]
    pub requested_bytes: i64,
    /// Total number of bytes allocated if known
    #[prost(int64, tag="2")]
    pub allocated_bytes: i64,
    /// Name of the allocator used
    #[prost(string, tag="3")]
    pub allocator_name: std::string::String,
    /// Identifier of the allocated buffer if known
    #[prost(int64, tag="4")]
    pub allocation_id: i64,
    /// Set if this tensor only has one remaining reference
    #[prost(bool, tag="5")]
    pub has_single_reference: bool,
    /// Address of the allocation.
    #[prost(uint64, tag="6")]
    pub ptr: u64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorDescription {
    /// Data type of tensor elements
    #[prost(enumeration="DataType", tag="1")]
    pub dtype: i32,
    /// Shape of the tensor.
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    /// Information about the size and allocator used for the data
    #[prost(message, optional, tag="4")]
    pub allocation_description: ::std::option::Option<AllocationDescription>,
}
/// An allocation/de-allocation operation performed by the allocator.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AllocationRecord {
    /// The timestamp of the operation.
    #[prost(int64, tag="1")]
    pub alloc_micros: i64,
    /// Number of bytes allocated, or de-allocated if negative.
    #[prost(int64, tag="2")]
    pub alloc_bytes: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AllocatorMemoryUsed {
    #[prost(string, tag="1")]
    pub allocator_name: std::string::String,
    /// These are per-node allocator memory stats.
    #[prost(int64, tag="2")]
    pub total_bytes: i64,
    #[prost(int64, tag="3")]
    pub peak_bytes: i64,
    /// The bytes that are not deallocated.
    #[prost(int64, tag="4")]
    pub live_bytes: i64,
    /// The allocation and deallocation timeline.
    #[prost(message, repeated, tag="6")]
    pub allocation_records: ::std::vec::Vec<AllocationRecord>,
    /// These are snapshots of the overall allocator memory stats.
    /// The number of live bytes currently allocated by the allocator.
    #[prost(int64, tag="5")]
    pub allocator_bytes_in_use: i64,
}
/// Output sizes recorded for a single execution of a graph node.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeOutput {
    #[prost(int32, tag="1")]
    pub slot: i32,
    #[prost(message, optional, tag="3")]
    pub tensor_description: ::std::option::Option<TensorDescription>,
}
/// For memory tracking.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryStats {
    #[prost(int64, tag="1")]
    pub temp_memory_size: i64,
    #[prost(int64, tag="3")]
    pub persistent_memory_size: i64,
    #[prost(int64, repeated, tag="5")]
    pub persistent_tensor_alloc_ids: ::std::vec::Vec<i64>,
    #[prost(int64, tag="2")]
    pub device_temp_memory_size: i64,
    #[prost(int64, tag="4")]
    pub device_persistent_memory_size: i64,
    #[prost(int64, repeated, packed="false", tag="6")]
    pub device_persistent_tensor_alloc_ids: ::std::vec::Vec<i64>,
}
/// Time/size stats recorded for a single execution of a graph node.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeExecStats {
    /// TODO(tucker): Use some more compact form of node identity than
    /// the full string name.  Either all processes should agree on a
    /// global id (cost_id?) for each node, or we should use a hash of
    /// the name.
    #[prost(string, tag="1")]
    pub node_name: std::string::String,
    #[prost(int64, tag="2")]
    pub all_start_micros: i64,
    #[prost(int64, tag="3")]
    pub op_start_rel_micros: i64,
    #[prost(int64, tag="4")]
    pub op_end_rel_micros: i64,
    #[prost(int64, tag="5")]
    pub all_end_rel_micros: i64,
    #[prost(message, repeated, tag="6")]
    pub memory: ::std::vec::Vec<AllocatorMemoryUsed>,
    #[prost(message, repeated, tag="7")]
    pub output: ::std::vec::Vec<NodeOutput>,
    #[prost(string, tag="8")]
    pub timeline_label: std::string::String,
    #[prost(int64, tag="9")]
    pub scheduled_micros: i64,
    #[prost(uint32, tag="10")]
    pub thread_id: u32,
    #[prost(message, repeated, tag="11")]
    pub referenced_tensor: ::std::vec::Vec<AllocationDescription>,
    #[prost(message, optional, tag="12")]
    pub memory_stats: ::std::option::Option<MemoryStats>,
    #[prost(int64, tag="13")]
    pub all_start_nanos: i64,
    #[prost(int64, tag="14")]
    pub op_start_rel_nanos: i64,
    #[prost(int64, tag="15")]
    pub op_end_rel_nanos: i64,
    #[prost(int64, tag="16")]
    pub all_end_rel_nanos: i64,
    #[prost(int64, tag="17")]
    pub scheduled_nanos: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceStepStats {
    #[prost(string, tag="1")]
    pub device: std::string::String,
    #[prost(message, repeated, tag="2")]
    pub node_stats: ::std::vec::Vec<NodeExecStats>,
    /// Its key is thread id.
    #[prost(map="uint32, string", tag="3")]
    pub thread_names: ::std::collections::HashMap<u32, std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StepStats {
    #[prost(message, repeated, tag="1")]
    pub dev_stats: ::std::vec::Vec<DeviceStepStats>,
}
/// Option for watching a node in TensorFlow Debugger (tfdbg).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebugTensorWatch {
    /// Name of the node to watch.
    /// Use "*" for wildcard. But note: currently, regex is not supported in
    /// general.
    #[prost(string, tag="1")]
    pub node_name: std::string::String,
    /// Output slot to watch.
    /// The semantics of output_slot == -1 is that all outputs of the node
    /// will be watched (i.e., a wildcard).
    /// Other negative values of output_slot are invalid and will lead to
    /// errors currently.
    #[prost(int32, tag="2")]
    pub output_slot: i32,
    /// Name(s) of the debugging op(s).
    /// One or more than one probes on a tensor.
    /// e.g., {"DebugIdentity", "DebugNanCount"}
    #[prost(string, repeated, tag="3")]
    pub debug_ops: ::std::vec::Vec<std::string::String>,
    /// URL(s) for debug targets(s).
    ///
    /// Supported URL formats are:
    ///   - file:///foo/tfdbg_dump: Writes out Event content to file
    ///     /foo/tfdbg_dump.  Assumes all directories can be created if they don't
    ///     already exist.
    ///   - grpc://localhost:11011: Sends an RPC request to an EventListener
    ///     service running at localhost:11011 with the event.
    ///   - memcbk:///event_key: Routes tensors to clients using the
    ///     callback registered with the DebugCallbackRegistry for event_key.
    ///
    /// Each debug op listed in debug_ops will publish its output tensor (debug
    /// signal) to all URLs in debug_urls.
    ///
    /// N.B. Session::Run() supports concurrent invocations of the same inputs
    /// (feed keys), outputs and target nodes. If such concurrent invocations
    /// are to be debugged, the callers of Session::Run() must use distinct
    /// debug_urls to make sure that the streamed or dumped events do not overlap
    /// among the invocations.
    /// TODO(cais): More visible documentation of this in g3docs.
    #[prost(string, repeated, tag="4")]
    pub debug_urls: ::std::vec::Vec<std::string::String>,
    /// Do not error out if debug op creation fails (e.g., due to dtype
    /// incompatibility). Instead, just log the failure.
    #[prost(bool, tag="5")]
    pub tolerate_debug_op_creation_failures: bool,
}
/// Options for initializing DebuggerState in TensorFlow Debugger (tfdbg).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebugOptions {
    /// Debugging options
    #[prost(message, repeated, tag="4")]
    pub debug_tensor_watch_opts: ::std::vec::Vec<DebugTensorWatch>,
    /// Caller-specified global step count.
    /// Note that this is distinct from the session run count and the executor
    /// step count.
    #[prost(int64, tag="10")]
    pub global_step: i64,
    /// Whether the total disk usage of tfdbg is to be reset to zero
    /// in this Session.run call. This is used by wrappers and hooks
    /// such as the local CLI ones to indicate that the dumped tensors
    /// are cleaned up from the disk after each Session.run.
    #[prost(bool, tag="11")]
    pub reset_disk_byte_usage: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebuggedSourceFile {
    /// The host name on which a source code file is located.
    #[prost(string, tag="1")]
    pub host: std::string::String,
    /// Path to the source code file.
    #[prost(string, tag="2")]
    pub file_path: std::string::String,
    /// The timestamp at which the source code file is last modified.
    #[prost(int64, tag="3")]
    pub last_modified: i64,
    /// Byte size of the file.
    #[prost(int64, tag="4")]
    pub bytes: i64,
    /// Line-by-line content of the source code file.
    #[prost(string, repeated, tag="5")]
    pub lines: ::std::vec::Vec<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebuggedSourceFiles {
    /// A collection of source code files.
    #[prost(message, repeated, tag="1")]
    pub source_files: ::std::vec::Vec<DebuggedSourceFile>,
}
/// Metadata associated with a series of Summary data
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummaryDescription {
    /// Hint on how plugins should process the data in this series.
    /// Supported values include "scalar", "histogram", "image", "audio"
    #[prost(string, tag="1")]
    pub type_hint: std::string::String,
}
/// Serialization format for histogram module in
/// core/lib/histogram/histogram.h
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HistogramProto {
    #[prost(double, tag="1")]
    pub min: f64,
    #[prost(double, tag="2")]
    pub max: f64,
    #[prost(double, tag="3")]
    pub num: f64,
    #[prost(double, tag="4")]
    pub sum: f64,
    #[prost(double, tag="5")]
    pub sum_squares: f64,
    /// Parallel arrays encoding the bucket boundaries and the bucket values.
    /// bucket(i) is the count for the bucket i.  The range for
    /// a bucket is:
    ///   i == 0:  -DBL_MAX .. bucket_limit(0)
    ///   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
    #[prost(double, repeated, tag="6")]
    pub bucket_limit: ::std::vec::Vec<f64>,
    #[prost(double, repeated, tag="7")]
    pub bucket: ::std::vec::Vec<f64>,
}
/// A SummaryMetadata encapsulates information on which plugins are able to make
/// use of a certain summary value.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummaryMetadata {
    /// Data that associates a summary with a certain plugin.
    #[prost(message, optional, tag="1")]
    pub plugin_data: ::std::option::Option<summary_metadata::PluginData>,
    /// Display name for viewing in TensorBoard.
    #[prost(string, tag="2")]
    pub display_name: std::string::String,
    /// Longform readable description of the summary sequence. Markdown supported.
    #[prost(string, tag="3")]
    pub summary_description: std::string::String,
}
pub mod summary_metadata {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct PluginData {
        /// The name of the plugin this data pertains to.
        #[prost(string, tag="1")]
        pub plugin_name: std::string::String,
        /// The content to store for the plugin. The best practice is for this to be
        /// a binary serialized protocol buffer.
        #[prost(bytes, tag="2")]
        pub content: std::vec::Vec<u8>,
    }
}
/// A Summary is a set of named values to be displayed by the
/// visualizer.
///
/// Summaries are produced regularly during training, as controlled by
/// the "summary_interval_secs" attribute of the training operation.
/// Summaries are also produced at the end of an evaluation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Summary {
    /// Set of values for the summary.
    #[prost(message, repeated, tag="1")]
    pub value: ::std::vec::Vec<summary::Value>,
}
pub mod summary {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Image {
        /// Dimensions of the image.
        #[prost(int32, tag="1")]
        pub height: i32,
        #[prost(int32, tag="2")]
        pub width: i32,
        /// Valid colorspace values are
        ///   1 - grayscale
        ///   2 - grayscale + alpha
        ///   3 - RGB
        ///   4 - RGBA
        ///   5 - DIGITAL_YUV
        ///   6 - BGRA
        #[prost(int32, tag="3")]
        pub colorspace: i32,
        /// Image data in encoded format.  All image formats supported by
        /// image_codec::CoderUtil can be stored here.
        #[prost(bytes, tag="4")]
        pub encoded_image_string: std::vec::Vec<u8>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Audio {
        /// Sample rate of the audio in Hz.
        #[prost(float, tag="1")]
        pub sample_rate: f32,
        /// Number of channels of audio.
        #[prost(int64, tag="2")]
        pub num_channels: i64,
        /// Length of the audio in frames (samples per channel).
        #[prost(int64, tag="3")]
        pub length_frames: i64,
        /// Encoded audio data and its associated RFC 2045 content type (e.g.
        /// "audio/wav").
        #[prost(bytes, tag="4")]
        pub encoded_audio_string: std::vec::Vec<u8>,
        #[prost(string, tag="5")]
        pub content_type: std::string::String,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Value {
        /// This field is deprecated and will not be set.
        #[prost(string, tag="7")]
        pub node_name: std::string::String,
        /// Tag name for the data. Used by TensorBoard plugins to organize data. Tags
        /// are often organized by scope (which contains slashes to convey
        /// hierarchy). For example: foo/bar/0
        #[prost(string, tag="1")]
        pub tag: std::string::String,
        /// Contains metadata on the summary value such as which plugins may use it.
        /// Take note that many summary values may lack a metadata field. This is
        /// because the FileWriter only keeps a metadata object on the first summary
        /// value with a certain tag for each tag. TensorBoard then remembers which
        /// tags are associated with which plugins. This saves space.
        #[prost(message, optional, tag="9")]
        pub metadata: ::std::option::Option<super::SummaryMetadata>,
        /// Value associated with the tag.
        #[prost(oneof="value::Value", tags="2, 3, 4, 5, 6, 8")]
        pub value: ::std::option::Option<value::Value>,
    }
    pub mod value {
        /// Value associated with the tag.
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Value {
            #[prost(float, tag="2")]
            SimpleValue(f32),
            #[prost(bytes, tag="3")]
            ObsoleteOldStyleHistogram(std::vec::Vec<u8>),
            #[prost(message, tag="4")]
            Image(super::Image),
            #[prost(message, tag="5")]
            Histo(super::super::HistogramProto),
            #[prost(message, tag="6")]
            Audio(super::Audio),
            #[prost(message, tag="8")]
            Tensor(super::super::TensorProto),
        }
    }
}
/// Protocol buffer representing an event that happened during
/// the execution of a Brain model.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Event {
    /// Timestamp of the event.
    #[prost(double, tag="1")]
    pub wall_time: f64,
    /// Global step of the event.
    #[prost(int64, tag="2")]
    pub step: i64,
    #[prost(oneof="event::What", tags="3, 4, 5, 6, 7, 8, 9")]
    pub what: ::std::option::Option<event::What>,
}
pub mod event {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum What {
        /// An event file was started, with the specified version.
        /// This is use to identify the contents of the record IO files
        /// easily.  Current version is "brain.Event:2".  All versions
        /// start with "brain.Event:".
        #[prost(string, tag="3")]
        FileVersion(std::string::String),
        /// An encoded version of a GraphDef.
        #[prost(bytes, tag="4")]
        GraphDef(std::vec::Vec<u8>),
        /// A summary was generated.
        #[prost(message, tag="5")]
        Summary(super::Summary),
        /// The user output a log message. Not all messages are logged, only ones
        /// generated via the Python tensorboard_logging module.
        #[prost(message, tag="6")]
        LogMessage(super::LogMessage),
        /// The state of the session which can be used for restarting after crashes.
        #[prost(message, tag="7")]
        SessionLog(super::SessionLog),
        /// The metadata returned by running a session.run() call.
        #[prost(message, tag="8")]
        TaggedRunMetadata(super::TaggedRunMetadata),
        /// An encoded version of a MetaGraphDef.
        #[prost(bytes, tag="9")]
        MetaGraphDef(std::vec::Vec<u8>),
    }
}
/// Protocol buffer used for logging messages to the events file.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LogMessage {
    #[prost(enumeration="log_message::Level", tag="1")]
    pub level: i32,
    #[prost(string, tag="2")]
    pub message: std::string::String,
}
pub mod log_message {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Level {
        Unknown = 0,
        /// Note: The logging level 10 cannot be named DEBUG. Some software
        /// projects compile their C/C++ code with -DDEBUG in debug builds. So the
        /// C++ code generated from this file should not have an identifier named
        /// DEBUG.
        Debugging = 10,
        Info = 20,
        Warn = 30,
        Error = 40,
        Fatal = 50,
    }
}
/// Protocol buffer used for logging session state.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SessionLog {
    #[prost(enumeration="session_log::SessionStatus", tag="1")]
    pub status: i32,
    /// This checkpoint_path contains both the path and filename.
    #[prost(string, tag="2")]
    pub checkpoint_path: std::string::String,
    #[prost(string, tag="3")]
    pub msg: std::string::String,
}
pub mod session_log {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum SessionStatus {
        StatusUnspecified = 0,
        Start = 1,
        Stop = 2,
        Checkpoint = 3,
    }
}
/// For logging the metadata output for a single session.run() call.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaggedRunMetadata {
    /// Tag name associated with this metadata.
    #[prost(string, tag="1")]
    pub tag: std::string::String,
    /// Byte-encoded version of the `RunMetadata` proto in order to allow lazy
    /// deserialization.
    #[prost(bytes, tag="2")]
    pub run_metadata: std::vec::Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WatchdogConfig {
    #[prost(int64, tag="1")]
    pub timeout_ms: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WorkerHeartbeatRequest {
    #[prost(enumeration="WorkerShutdownMode", tag="1")]
    pub shutdown_mode: i32,
    #[prost(message, optional, tag="2")]
    pub watchdog_config: ::std::option::Option<WatchdogConfig>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WorkerHeartbeatResponse {
    #[prost(enumeration="WorkerHealth", tag="1")]
    pub health_status: i32,
    #[prost(message, repeated, tag="2")]
    pub worker_log: ::std::vec::Vec<Event>,
    #[prost(string, tag="3")]
    pub hostname: std::string::String,
}
// Worker heartbeat messages.  Support for these operations is currently
// internal and expected to change.

/// Current health status of a worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum WorkerHealth {
    /// By default a worker is healthy.
    Ok = 0,
    ReceivedShutdownSignal = 1,
    InternalError = 2,
    /// Worker has been instructed to shutdown after a timeout.
    ShuttingDown = 3,
}
/// Indicates the behavior of the worker when an internal error or shutdown
/// signal is received.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum WorkerShutdownMode {
    Default = 0,
    NotConfigured = 1,
    WaitForCoordinator = 2,
    ShutdownAfterTimeout = 3,
}
/// Reply message from EventListener to the client, i.e., to the source of the
/// Event protocol buffers, e.g., debug ops inserted by a debugged runtime to a
/// TensorFlow graph being executed.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EventReply {
    #[prost(message, repeated, tag="1")]
    pub debug_op_state_changes: ::std::vec::Vec<event_reply::DebugOpStateChange>,
    /// New tensor value to override the current tensor value with.
    ///
    /// TODO(cais): Make use of this field to implement overriding of tensor value
    /// during debugging.
    #[prost(message, optional, tag="2")]
    pub tensor: ::std::option::Option<TensorProto>,
}
pub mod event_reply {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DebugOpStateChange {
        #[prost(enumeration="debug_op_state_change::State", tag="1")]
        pub state: i32,
        #[prost(string, tag="2")]
        pub node_name: std::string::String,
        #[prost(int32, tag="3")]
        pub output_slot: i32,
        #[prost(string, tag="4")]
        pub debug_op: std::string::String,
    }
    pub mod debug_op_state_change {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
        #[repr(i32)]
        pub enum State {
            Unspecified = 0,
            Disabled = 1,
            ReadOnly = 2,
            ReadWrite = 3,
        }
    }
}
/// Data on the traceback of a debugged call, e.g., a Session.run() call, or the
/// execution of an eager operation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CallTraceback {
    #[prost(enumeration="call_traceback::CallType", tag="1")]
    pub call_type: i32,
    /// A key for the call. For example, for graph execution, this is a key
    /// consisting of the names of the fed and fetched tensors.
    #[prost(string, tag="2")]
    pub call_key: std::string::String,
    /// Traceback stack for the origin of the call event.
    /// For graph execution, this is the stack of the Session.run() call.
    /// For eager execution, this is the stack of the Python line that invokes
    /// the execution of the eager op.
    #[prost(message, optional, tag="3")]
    pub origin_stack: ::std::option::Option<tfprof::CodeDef>,
    /// Keeps track of the mapping from integer IDs in `origin_stack` to actual
    /// string values (e.g., file paths, function names).
    #[prost(map="int64, string", tag="4")]
    pub origin_id_to_string: ::std::collections::HashMap<i64, std::string::String>,
    /// Traceback for the graph (if any) involved in the call.
    #[prost(message, optional, tag="5")]
    pub graph_traceback: ::std::option::Option<tfprof::OpLogProto>,
    /// Version of the graph in `graph_traceback` (if any).
    #[prost(int64, tag="6")]
    pub graph_version: i64,
}
pub mod call_traceback {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum CallType {
        Unspecified = 0,
        GraphExecution = 1,
        EagerExecution = 2,
    }
}
/// A message that describes one region of memmapped file.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemmappedFileSystemDirectoryElement {
    #[prost(uint64, tag="1")]
    pub offset: u64,
    #[prost(string, tag="2")]
    pub name: std::string::String,
    #[prost(uint64, tag="3")]
    pub length: u64,
}
/// A directory of regions in a memmapped file.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemmappedFileSystemDirectory {
    #[prost(message, repeated, tag="1")]
    pub element: ::std::vec::Vec<MemmappedFileSystemDirectoryElement>,
}
/// Can only be interpreted if you know the corresponding TensorShape.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorSliceProto {
    /// Extent of the slice in all tensor dimensions.
    ///
    /// Must have one entry for each of the dimension of the tensor that this
    /// slice belongs to.  The order of sizes is the same as the order of
    /// dimensions in the TensorShape.
    #[prost(message, repeated, tag="1")]
    pub extent: ::std::vec::Vec<tensor_slice_proto::Extent>,
}
pub mod tensor_slice_proto {
    /// Extent of the slice in one dimension.
    ///
    /// Either both or no attributes must be set.  When no attribute is set
    /// means: All data in that dimension.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Extent {
        /// Start index of the slice, starting at 0.
        #[prost(int64, tag="1")]
        pub start: i64,
        /// Length of the slice: if the length is missing or -1 we will
        /// interpret this as "everything in this dimension".  We use
        /// "oneof" to preserve information about whether the length is
        /// present without changing the serialization format from the
        /// prior proto2 version of this proto.
        #[prost(oneof="extent::HasLength", tags="2")]
        pub has_length: ::std::option::Option<extent::HasLength>,
    }
    pub mod extent {
        /// Length of the slice: if the length is missing or -1 we will
        /// interpret this as "everything in this dimension".  We use
        /// "oneof" to preserve information about whether the length is
        /// present without changing the serialization format from the
        /// prior proto2 version of this proto.
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum HasLength {
            #[prost(int64, tag="2")]
            Length(i64),
        }
    }
}
/// Version information for a piece of serialized data
///
/// There are different types of versions for each type of data
/// (GraphDef, etc.), but they all have the same common shape
/// described here.
///
/// Each consumer has "consumer" and "min_producer" versions (specified
/// elsewhere).  A consumer is allowed to consume this data if
///
///   producer >= min_producer
///   consumer >= min_consumer
///   consumer not in bad_consumers
///
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VersionDef {
    /// The version of the code that produced this data.
    #[prost(int32, tag="1")]
    pub producer: i32,
    /// Any consumer below this version is not allowed to consume this data.
    #[prost(int32, tag="2")]
    pub min_consumer: i32,
    /// Specific consumer versions which are disallowed (e.g. due to bugs).
    #[prost(int32, repeated, tag="3")]
    pub bad_consumers: ::std::vec::Vec<i32>,
}
/// Metadata describing the set of slices of the same tensor saved in a
/// checkpoint file.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedSliceMeta {
    /// Name of the tensor.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// Shape of the tensor
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    /// Type of the tensor
    #[prost(enumeration="DataType", tag="3")]
    pub r#type: i32,
    /// Explicit list of slices saved in the checkpoint file.
    #[prost(message, repeated, tag="4")]
    pub slice: ::std::vec::Vec<TensorSliceProto>,
}
/// Metadata describing the set of tensor slices saved in a checkpoint file.
/// It is always stored at the beginning of each checkpoint file.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedTensorSliceMeta {
    /// Each SavedSliceMeta describes the slices for one tensor.
    #[prost(message, repeated, tag="1")]
    pub tensor: ::std::vec::Vec<SavedSliceMeta>,
    /// Compatibility version of this checkpoint.  See core/public/version.h
    /// for version history.
    #[prost(message, optional, tag="2")]
    pub versions: ::std::option::Option<VersionDef>,
}
/// Saved tensor slice: it stores the name of the tensors, the slice, and the
/// raw data.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedSlice {
    /// Name of the tensor that this slice belongs to. This must be identical to
    /// the name used to encode the key for this record.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// Extent of the slice.  Must have one entry for each of the dimension of the
    /// tensor that this slice belongs to.
    #[prost(message, optional, tag="2")]
    pub slice: ::std::option::Option<TensorSliceProto>,
    /// The raw data of the slice is stored as a TensorProto. Only raw data are
    /// stored (we don't fill in fields such as dtype or tensor_shape).
    #[prost(message, optional, tag="3")]
    pub data: ::std::option::Option<TensorProto>,
}
/// Each record in a v3 checkpoint file is a serialized SavedTensorSlices
/// message.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedTensorSlices {
    /// This is only present at the first item of each checkpoint file and serves
    /// as a table of contents, listing all the tensor slices saved in this file.
    #[prost(message, optional, tag="1")]
    pub meta: ::std::option::Option<SavedTensorSliceMeta>,
    /// This exists in all but the first item of each checkpoint file.
    #[prost(message, optional, tag="2")]
    pub data: ::std::option::Option<SavedSlice>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeDef {
    /// The name given to this operator. Used for naming inputs,
    /// logging, visualization, etc.  Unique within a single GraphDef.
    /// Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_>./]*".
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The operation name.  There may be custom parameters in attrs.
    /// Op names starting with an underscore are reserved for internal use.
    #[prost(string, tag="2")]
    pub op: std::string::String,
    /// Each input is "node:src_output" with "node" being a string name and
    /// "src_output" indicating which output tensor to use from "node". If
    /// "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
    /// may optionally be followed by control inputs that have the format
    /// "^node".
    #[prost(string, repeated, tag="3")]
    pub input: ::std::vec::Vec<std::string::String>,
    /// A (possibly partial) specification for the device on which this
    /// node should be placed.
    /// The expected syntax for this string is as follows:
    ///
    /// DEVICE_SPEC ::= PARTIAL_SPEC
    ///
    /// PARTIAL_SPEC ::= ("/" CONSTRAINT) *
    /// CONSTRAINT ::= ("job:" JOB_NAME)
    ///              | ("replica:" [1-9][0-9]*)
    ///              | ("task:" [1-9][0-9]*)
    ///              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
    ///
    /// Valid values for this string include:
    /// * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
    /// * "/job:worker/device:GPU:3"                   (partial specification)
    /// * ""                                    (no specification)
    ///
    /// If the constraints do not resolve to a single device (or if this
    /// field is empty or not present), the runtime will attempt to
    /// choose a device automatically.
    #[prost(string, tag="4")]
    pub device: std::string::String,
    /// Operation-specific graph-construction-time configuration.
    /// Note that this should include all attrs defined in the
    /// corresponding OpDef, including those with a value matching
    /// the default -- this allows the default to change and makes
    /// NodeDefs easier to interpret on their own.  However, if
    /// an attr with a default is not specified in this list, the
    /// default will be used.
    /// The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
    /// one of the names from the corresponding OpDef's attr field).
    /// The values must have a type matching the corresponding OpDef
    /// attr's type field.
    /// TODO(josh11b): Add some examples here showing best practices.
    #[prost(map="string, message", tag="5")]
    pub attr: ::std::collections::HashMap<std::string::String, AttrValue>,
    /// This stores debug information associated with the node.
    #[prost(message, optional, tag="6")]
    pub experimental_debug_info: ::std::option::Option<node_def::ExperimentalDebugInfo>,
}
pub mod node_def {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ExperimentalDebugInfo {
        /// Opaque string inserted into error messages created by the runtime.
        ///
        /// This is intended to store the list of names of the nodes from the
        /// original graph that this node was derived. For example if this node, say
        /// C, was result of a fusion of 2 nodes A and B, then 'original_node' would
        /// be {A, B}. This information can be used to map errors originating at the
        /// current node to some top level source code.
        #[prost(string, repeated, tag="1")]
        pub original_node_names: ::std::vec::Vec<std::string::String>,
        /// This is intended to store the list of names of the functions from the
        /// original graph that this node was derived. For example if this node, say
        /// C, was result of a fusion of node A in function FA and node B in function
        /// FB, then `original_funcs` would be {FA, FB}. If the node is in the top
        /// level graph, the `original_func` is empty. This information, with the
        /// `original_node_names` can be used to map errors originating at the
        /// current ndoe to some top level source code.
        #[prost(string, repeated, tag="2")]
        pub original_func_names: ::std::vec::Vec<std::string::String>,
    }
}
/// Defines an operation. A NodeDef in a GraphDef specifies an Op by
/// using the "op" field which should match the name of a OpDef.
/// LINT.IfChange
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpDef {
    /// Op names starting with an underscore are reserved for internal use.
    /// Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9>_]*".
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// Description of the input(s).
    #[prost(message, repeated, tag="2")]
    pub input_arg: ::std::vec::Vec<op_def::ArgDef>,
    /// Description of the output(s).
    #[prost(message, repeated, tag="3")]
    pub output_arg: ::std::vec::Vec<op_def::ArgDef>,
    /// Named control outputs for this operation. Useful only for composite
    /// operations (i.e. functions) which want to name different control outputs.
    #[prost(string, repeated, tag="20")]
    pub control_output: ::std::vec::Vec<std::string::String>,
    #[prost(message, repeated, tag="4")]
    pub attr: ::std::vec::Vec<op_def::AttrDef>,
    /// Optional deprecation based on GraphDef versions.
    #[prost(message, optional, tag="8")]
    pub deprecation: ::std::option::Option<OpDeprecation>,
    /// One-line human-readable description of what the Op does.
    #[prost(string, tag="5")]
    pub summary: std::string::String,
    /// Additional, longer human-readable description of what the Op does.
    #[prost(string, tag="6")]
    pub description: std::string::String,
    // -------------------------------------------------------------------------
    // Which optimizations this operation can participate in.

    /// True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
    #[prost(bool, tag="18")]
    pub is_commutative: bool,
    /// If is_aggregate is true, then this operation accepts N >= 2
    /// inputs and produces 1 output all of the same type.  Should be
    /// associative and commutative, and produce output with the same
    /// shape as the input.  The optimizer may replace an aggregate op
    /// taking input from multiple devices with a tree of aggregate ops
    /// that aggregate locally within each device (and possibly within
    /// groups of nearby devices) before communicating.
    /// TODO(josh11b): Implement that optimization.
    ///
    /// for things like add
    #[prost(bool, tag="16")]
    pub is_aggregate: bool,
    // Other optimizations go here, like
    //   can_alias_input, rewrite_when_output_unused, partitioning_strategy, etc.

    // -------------------------------------------------------------------------
    // Optimization constraints.

    /// Ops are marked as stateful if their behavior depends on some state beyond
    /// their input tensors (e.g. variable reading op) or if they have
    /// a side-effect (e.g. printing or asserting ops). Equivalently, stateless ops
    /// must always produce the same output for the same input and have
    /// no side-effects.
    ///
    /// By default Ops may be moved between devices.  Stateful ops should
    /// either not be moved, or should only be moved if that state can also
    /// be moved (e.g. via some sort of save / restore).
    /// Stateful ops are guaranteed to never be optimized away by Common
    /// Subexpression Elimination (CSE).
    ///
    /// for things like variables, queue
    #[prost(bool, tag="17")]
    pub is_stateful: bool,
    // -------------------------------------------------------------------------
    // Non-standard options.

    /// By default, all inputs to an Op must be initialized Tensors.  Ops
    /// that may initialize tensors for the first time should set this
    /// field to true, to allow the Op to take an uninitialized Tensor as
    /// input.
    ///
    /// for Assign, etc.
    #[prost(bool, tag="19")]
    pub allows_uninitialized_input: bool,
}
pub mod op_def {
    /// For describing inputs and outputs.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ArgDef {
        /// Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// Human readable description.
        #[prost(string, tag="2")]
        pub description: std::string::String,
        /// Describes the type of one or more tensors that are accepted/produced
        /// by this input/output arg.  The only legal combinations are:
        /// * For a single tensor: either the "type" field is set or the
        ///   "type_attr" field is set to the name of an attr with type "type".
        /// * For a sequence of tensors with the same type: the "number_attr"
        ///   field will be set to the name of an attr with type "int", and
        ///   either the "type" or "type_attr" field will be set as for
        ///   single tensors.
        /// * For a sequence of tensors, the "type_list_attr" field will be set
        ///   to the name of an attr with type "list(type)".
        #[prost(enumeration="super::DataType", tag="3")]
        pub r#type: i32,
        /// if specified, attr must have type "type"
        #[prost(string, tag="4")]
        pub type_attr: std::string::String,
        /// if specified, attr must have type "int"
        #[prost(string, tag="5")]
        pub number_attr: std::string::String,
        /// If specified, attr must have type "list(type)", and none of
        /// type, type_attr, and number_attr may be specified.
        #[prost(string, tag="6")]
        pub type_list_attr: std::string::String,
        /// For inputs: if true, the inputs are required to be refs.
        ///   By default, inputs can be either refs or non-refs.
        /// For outputs: if true, outputs are refs, otherwise they are not.
        #[prost(bool, tag="16")]
        pub is_ref: bool,
    }
    /// Description of the graph-construction-time configuration of this
    /// Op.  That is to say, this describes the attr fields that will
    /// be specified in the NodeDef.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AttrDef {
        /// A descriptive name for the argument.  May be used, e.g. by the
        /// Python client, as a keyword argument name, and so should match
        /// the regexp "[a-z][a-z0-9_]+".
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// One of the type names from attr_value.proto ("string", "list(string)",
        /// "int", etc.).
        #[prost(string, tag="2")]
        pub r#type: std::string::String,
        /// A reasonable default for this attribute if the user does not supply
        /// a value.  If not specified, the user must supply a value.
        #[prost(message, optional, tag="3")]
        pub default_value: ::std::option::Option<super::AttrValue>,
        /// Human-readable description.
        #[prost(string, tag="4")]
        pub description: std::string::String,
        // TODO(josh11b): bool is_optional?

        // --- Constraints ---
        // These constraints are only in effect if specified.  Default is no
        // constraints.

        /// For type == "int", this is a minimum value.  For "list(___)"
        /// types, this is the minimum length.
        #[prost(bool, tag="5")]
        pub has_minimum: bool,
        #[prost(int64, tag="6")]
        pub minimum: i64,
        /// The set of allowed values.  Has type that is the "list" version
        /// of the "type" field above (uses the "list" field of AttrValue).
        /// If type == "type" or "list(type)" above, then the "type" field
        /// of "allowed_values.list" has the set of allowed DataTypes.
        /// If type == "string" or "list(string)", then the "s" field of
        /// "allowed_values.list" has the set of allowed strings.
        #[prost(message, optional, tag="7")]
        pub allowed_values: ::std::option::Option<super::AttrValue>,
    }
}
/// Information about version-dependent deprecation of an op
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpDeprecation {
    /// First GraphDef version at which the op is disallowed.
    #[prost(int32, tag="1")]
    pub version: i32,
    /// Explanation of why it was deprecated and what to use instead.
    #[prost(string, tag="2")]
    pub explanation: std::string::String,
}
/// A collection of OpDefs
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpList {
    #[prost(message, repeated, tag="1")]
    pub op: ::std::vec::Vec<OpDef>,
}
/// A library is a set of named functions.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FunctionDefLibrary {
    #[prost(message, repeated, tag="1")]
    pub function: ::std::vec::Vec<FunctionDef>,
    #[prost(message, repeated, tag="2")]
    pub gradient: ::std::vec::Vec<GradientDef>,
}
/// A function can be instantiated when the runtime can bind every attr
/// with a value. When a GraphDef has a call to a function, it must
/// have binding for every attr defined in the signature.
///
/// TODO(zhifengc):
///   * device spec, etc.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FunctionDef {
    /// The definition of the function's name, arguments, return values,
    /// attrs etc.
    #[prost(message, optional, tag="1")]
    pub signature: ::std::option::Option<OpDef>,
    /// Attributes specific to this function definition.
    #[prost(map="string, message", tag="5")]
    pub attr: ::std::collections::HashMap<std::string::String, AttrValue>,
    #[prost(map="uint32, message", tag="7")]
    pub arg_attr: ::std::collections::HashMap<u32, function_def::ArgAttrs>,
    // In both of the following fields, there is the need to specify an
    // output that is used as either the input to another node (in
    // `node_def`) or as a return value of the function (in `ret`).
    // Unlike the NodeDefs in GraphDef, we need to be able to specify a
    // list in some cases (instead of just single outputs).  Also, we
    // need to be able to deal with lists of unknown length (so the
    // output index may not be known at function definition time).  So
    // we use the following format instead:
    // * "fun_in" where "fun_in" is the name of a function input arg in
    //   the `signature` field above.  This represents that input, whether
    //   it is a single tensor or a list.
    // * "fun_in:0" gives the first element of a function input arg (a
    //   non-list input is considered a list of length 1 for these
    //   purposes).
    // * "node:out" where "node" is the name of a node in `node_def` and
    //   "out" is the name one of its op's output arguments (the name
    //   comes from the OpDef of the node's op). This represents that
    //   node's output, whether it is a single tensor or a list.
    //   Note: We enforce that an op's output arguments are never
    //   renamed in the backwards-compatibility test.
    // * "node:out:0" gives the first element of a node output arg (a
    //   non-list output is considered a list of length 1 for these
    //   purposes).
    //
    // NOT CURRENTLY SUPPORTED (but may be in the future):
    // * "node:out:-1" gives last element in a node output list
    // * "node:out:1:" gives a list with all but the first element in a
    //   node output list
    // * "node:out::-1" gives a list with all but the last element in a
    //   node output list

    // The body of the function.  Unlike the NodeDefs in a GraphDef, attrs
    // may have values of type `placeholder` and the `input` field uses
    // the "output" format above.

    /// By convention, "op" in node_def is resolved by consulting with a
    /// user-defined library first. If not resolved, "func" is assumed to
    /// be a builtin op.
    #[prost(message, repeated, tag="3")]
    pub node_def: ::std::vec::Vec<NodeDef>,
    /// A mapping from the output arg names from `signature` to the
    /// outputs from `node_def` that should be returned by the function.
    #[prost(map="string, string", tag="4")]
    pub ret: ::std::collections::HashMap<std::string::String, std::string::String>,
    /// A mapping from control output names from `signature` to node names in
    /// `node_def` which should be control outputs of this function.
    #[prost(map="string, string", tag="6")]
    pub control_ret: ::std::collections::HashMap<std::string::String, std::string::String>,
}
pub mod function_def {
    /// Attributes for function arguments. These attributes are the same set of
    /// valid attributes as to _Arg nodes.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ArgAttrs {
        #[prost(map="string, message", tag="1")]
        pub attr: ::std::collections::HashMap<std::string::String, super::AttrValue>,
    }
}
/// GradientDef defines the gradient function of a function defined in
/// a function library.
///
/// A gradient function g (specified by gradient_func) for a function f
/// (specified by function_name) must follow the following:
///
/// The function 'f' must be a numerical function which takes N inputs
/// and produces M outputs. Its gradient function 'g', which is a
/// function taking N + M inputs and produces N outputs.
///
/// I.e. if we have
///    (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
/// then, g is
///    (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
///                                      dL/dy1, dL/dy2, ..., dL/dy_M),
/// where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
/// loss function). dL/dx_i is the partial derivative of L with respect
/// to x_i.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GradientDef {
    /// The function name.
    #[prost(string, tag="1")]
    pub function_name: std::string::String,
    /// The gradient function's name.
    #[prost(string, tag="2")]
    pub gradient_func: std::string::String,
}
/// Represents the graph of operations
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphDef {
    #[prost(message, repeated, tag="1")]
    pub node: ::std::vec::Vec<NodeDef>,
    /// Compatibility versions of the graph.  See core/public/version.h for version
    /// history.  The GraphDef version is distinct from the TensorFlow version, and
    /// each release of TensorFlow will support a range of GraphDef versions.
    #[prost(message, optional, tag="4")]
    pub versions: ::std::option::Option<VersionDef>,
    /// Deprecated single version field; use versions above instead.  Since all
    /// GraphDef changes before "versions" was introduced were forward
    /// compatible, this field is entirely ignored.
    #[prost(int32, tag="3")]
    pub version: i32,
    /// EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
    ///
    /// "library" provides user-defined functions.
    ///
    /// Naming:
    ///   * library.function.name are in a flat namespace.
    ///     NOTE: We may need to change it to be hierarchical to support
    ///     different orgs. E.g.,
    ///     { "/google/nn", { ... }},
    ///     { "/google/vision", { ... }}
    ///     { "/org_foo/module_bar", { ... }}
    ///     map<string, FunctionDefLib> named_lib;
    ///   * If node[i].op is the name of one function in "library",
    ///     node[i] is deemed as a function call. Otherwise, node[i].op
    ///     must be a primitive operation supported by the runtime.
    ///
    ///
    /// Function call semantics:
    ///
    ///   * The callee may start execution as soon as some of its inputs
    ///     are ready. The caller may want to use Tuple() mechanism to
    ///     ensure all inputs are ready in the same time.
    ///
    ///   * The consumer of return values may start executing as soon as
    ///     the return values the consumer depends on are ready.  The
    ///     consumer may want to use Tuple() mechanism to ensure the
    ///     consumer does not start until all return values of the callee
    ///     function are ready.
    #[prost(message, optional, tag="2")]
    pub library: ::std::option::Option<FunctionDefLibrary>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CostGraphDef {
    #[prost(message, repeated, tag="1")]
    pub node: ::std::vec::Vec<cost_graph_def::Node>,
}
pub mod cost_graph_def {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Node {
        /// The name of the node. Names are globally unique.
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// The device of the node. Can be empty if the node is mapped to the
        /// default partition or partitioning hasn't been run yet.
        #[prost(string, tag="2")]
        pub device: std::string::String,
        /// The id of the node. Node ids are only unique inside a partition.
        #[prost(int32, tag="3")]
        pub id: i32,
        #[prost(message, repeated, tag="4")]
        pub input_info: ::std::vec::Vec<node::InputInfo>,
        #[prost(message, repeated, tag="5")]
        pub output_info: ::std::vec::Vec<node::OutputInfo>,
        /// Temporary memory used by this node.
        #[prost(int64, tag="6")]
        pub temporary_memory_size: i64,
        /// Persistent memory used by this node.
        #[prost(int64, tag="12")]
        pub persistent_memory_size: i64,
        #[prost(int64, tag="10")]
        pub host_temp_memory_size: i64,
        #[prost(int64, tag="11")]
        pub device_temp_memory_size: i64,
        #[prost(int64, tag="16")]
        pub device_persistent_memory_size: i64,
        /// Estimate of the computational cost of this node, in microseconds.
        #[prost(int64, tag="9")]
        pub compute_cost: i64,
        /// Analytical estimate of the computational cost of this node, in
        /// microseconds.
        #[prost(int64, tag="14")]
        pub compute_time: i64,
        /// Analytical estimate of the memory access cost of this node, in
        /// microseconds.
        #[prost(int64, tag="15")]
        pub memory_time: i64,
        /// If true, the output is permanent: it can't be discarded, because this
        /// node is part of the "final output". Nodes may depend on final nodes.
        #[prost(bool, tag="7")]
        pub is_final: bool,
        /// Ids of the control inputs for this node.
        #[prost(int32, repeated, tag="8")]
        pub control_input: ::std::vec::Vec<i32>,
        /// Are the costs inaccurate?
        #[prost(bool, tag="17")]
        pub inaccurate: bool,
    }
    pub mod node {
        /// Inputs of this node. They must be executed before this node can be
        /// executed. An input is a particular output of another node, specified
        /// by the node id and the output index.
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct InputInfo {
            #[prost(int32, tag="1")]
            pub preceding_node: i32,
            #[prost(int32, tag="2")]
            pub preceding_port: i32,
        }
        /// Outputs of this node.
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct OutputInfo {
            #[prost(int64, tag="1")]
            pub size: i64,
            /// If >= 0, the output is an alias of an input. Note that an alias input
            /// may itself be an alias. The algorithm will therefore need to follow
            /// those pointers.
            #[prost(int64, tag="2")]
            pub alias_input_port: i64,
            #[prost(message, optional, tag="3")]
            pub shape: ::std::option::Option<super::super::TensorShapeProto>,
            #[prost(enumeration="super::super::DataType", tag="4")]
            pub dtype: i32,
        }
    }
}
// This file contains protos to be used when defining a TensorFlow
// cluster.
//
// EXAMPLES
// --------
//
// 1. A single-process cluster, containing "/job:local/task:0".
//
//    Cluster:
//      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
//
//    Server:
//      cluster { $CLUSTER } job_name: 'local' task_index: 0
//
// 2. A two-process cluster, containing "/job:local/task:{0,1}".
//
//    Cluster:
//      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
//                          tasks { key: 1 value: 'localhost:2223' } }
//
//    Servers:
//      cluster { $CLUSTER } job_name: 'local' task_index: 0
//      cluster { $CLUSTER } job_name: 'local' task_index: 1
//
// 3. A two-job cluster, containing "/job:worker/task:{0,1,2}" and
//    "/job:ps/task:{0,1}".
//
//    Cluster:
//      job { name: 'worker' tasks { key: 0 value: 'worker1:2222' }
//                           tasks { key: 1 value: 'worker2:2222' }
//                           tasks { key: 2 value: 'worker3:2222' } }
//      job { name: 'ps'     tasks { key: 0 value: 'ps0:2222' }
//                           tasks { key: 1 value: 'ps1:2222' } }
//
//    Servers:
//      cluster { $CLUSTER } job_name: 'worker' task_index: 0
//      cluster { $CLUSTER } job_name: 'worker' task_index: 1
//      cluster { $CLUSTER } job_name: 'worker' task_index: 2
//      cluster { $CLUSTER } job_name: 'ps'     task_index: 0
//      cluster { $CLUSTER } job_name: 'ps'     task_index: 1

/// Defines a single job in a TensorFlow cluster.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct JobDef {
    /// The name of this job.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// Mapping from task ID to "hostname:port" string.
    ///
    /// If the `name` field contains "worker", and the `tasks` map contains a
    /// mapping from 7 to "example.org:2222", then the device prefix
    /// "/job:worker/task:7" will be assigned to "example.org:2222".
    #[prost(map="int32, string", tag="2")]
    pub tasks: ::std::collections::HashMap<i32, std::string::String>,
}
/// Defines a TensorFlow cluster as a set of jobs.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClusterDef {
    /// The jobs that comprise the cluster.
    #[prost(message, repeated, tag="1")]
    pub job: ::std::vec::Vec<JobDef>,
}
/// The config for graph verifiers.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VerifierConfig {
    /// Deadline for completion of all verification i.e. all the Toggle ON
    /// verifiers must complete execution within this time.
    #[prost(int64, tag="1")]
    pub verification_timeout_in_ms: i64,
    /// Perform structural validation on a tensorflow graph. Default is OFF.
    #[prost(enumeration="verifier_config::Toggle", tag="2")]
    pub structure_verifier: i32,
}
pub mod verifier_config {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Toggle {
        Default = 0,
        On = 1,
        Off = 2,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AutoParallelOptions {
    #[prost(bool, tag="1")]
    pub enable: bool,
    #[prost(int32, tag="2")]
    pub num_replicas: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ScopedAllocatorOptions {
    /// If present, only perform optimization for these ops.
    #[prost(string, repeated, tag="1")]
    pub enable_op: ::std::vec::Vec<std::string::String>,
}
/// Graph rewriting is experimental and subject to change, not covered by any
/// API stability guarantees.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RewriterConfig {
    /// Optimize tensor layouts (default is ON)
    /// e.g. This will try to use NCHW layout on GPU which is faster.
    #[prost(enumeration="rewriter_config::Toggle", tag="1")]
    pub layout_optimizer: i32,
    /// Fold constants (default is ON)
    /// Statically infer the value of tensors when possible, and materialize the
    /// result using constants.
    #[prost(enumeration="rewriter_config::Toggle", tag="3")]
    pub constant_folding: i32,
    /// Shape optimizations (default is ON)
    /// Simplify computations made on shapes.
    #[prost(enumeration="rewriter_config::Toggle", tag="13")]
    pub shape_optimization: i32,
    /// Remapping (default is ON)
    /// Remap subgraphs onto more efficient implementations.
    #[prost(enumeration="rewriter_config::Toggle", tag="14")]
    pub remapping: i32,
    /// Arithmetic optimizations (default is ON)
    /// e.g. Simplify arithmetic ops; merge ops with same value (like constants).
    #[prost(enumeration="rewriter_config::Toggle", tag="7")]
    pub arithmetic_optimization: i32,
    /// Control dependency optimizations (default is ON).
    /// Remove redundant control dependencies, which may enable other optimization.
    #[prost(enumeration="rewriter_config::Toggle", tag="8")]
    pub dependency_optimization: i32,
    /// Loop optimizations (default is ON).
    #[prost(enumeration="rewriter_config::Toggle", tag="9")]
    pub loop_optimization: i32,
    /// Function optimizations (default is ON).
    #[prost(enumeration="rewriter_config::Toggle", tag="10")]
    pub function_optimization: i32,
    /// Strips debug-related nodes from the graph (off by default).
    #[prost(enumeration="rewriter_config::Toggle", tag="11")]
    pub debug_stripper: i32,
    /// If true, don't remove unnecessary ops from the graph
    #[prost(bool, tag="2")]
    pub disable_model_pruning: bool,
    /// Try to allocate some independent Op outputs contiguously in order to
    /// merge or eliminate downstream Ops (off by default).
    #[prost(enumeration="rewriter_config::Toggle", tag="15")]
    pub scoped_allocator_optimization: i32,
    /// Force small ops onto the CPU (default is OFF).
    #[prost(enumeration="rewriter_config::Toggle", tag="18")]
    pub pin_to_host_optimization: i32,
    /// Enable the swap of kernel implementations based on the device placement
    /// (default is ON).
    #[prost(enumeration="rewriter_config::Toggle", tag="22")]
    pub implementation_selector: i32,
    /// Optimize data types (default is OFF).
    /// e.g., This will try to use float16 on GPU which is faster.
    /// Note that this can change the numerical stability of the graph and may
    /// require the use of loss scaling to maintain model convergence.
    #[prost(enumeration="rewriter_config::Toggle", tag="23")]
    pub auto_mixed_precision: i32,
    /// Disable the entire meta optimizer (off by default).
    #[prost(bool, tag="19")]
    pub disable_meta_optimizer: bool,
    /// Controls how many times we run the optimizers in meta optimizer (default
    /// is once).
    #[prost(enumeration="rewriter_config::NumIterationsType", tag="12")]
    pub meta_optimizer_iterations: i32,
    /// The minimum number of nodes in a graph to optimizer. For smaller graphs,
    /// optimization is skipped.
    /// 0 means the system picks an appropriate number.
    /// < 0 means do not skip optimization.
    #[prost(int32, tag="17")]
    pub min_graph_nodes: i32,
    /// Configures memory optimization passes through the meta-optimizer. Has no
    /// effect on manually requested memory optimization passes in the optimizers
    /// field.
    #[prost(enumeration="rewriter_config::MemOptType", tag="4")]
    pub memory_optimization: i32,
    /// A node name scope for node names which are valid outputs of recompuations.
    /// Inputs to nodes that match this scope may be recomputed (subject either to
    /// manual annotation of those input nodes or to manual annotation and
    /// heuristics depending on memory_optimization), but the nodes themselves will
    /// not be recomputed. This matches any sub-scopes as well, meaning the scope
    /// can appear not just as a top-level scope. For example, if the value is
    /// "gradients/", the default, it will match node name "gradients/foo",
    /// "foo/gradients/bar", but not "foo_gradients/"
    #[prost(string, tag="6")]
    pub memory_optimizer_target_node_name_scope: std::string::String,
    /// Maximum number of milliseconds to spend optimizing a single graph before
    /// timing out. If equal to 0 the system picks a default (currently 5 minutes).
    /// If less than 0 the optimizer will never time out.
    #[prost(int64, tag="20")]
    pub meta_optimizer_timeout_ms: i64,
    /// Configures AutoParallel optimization passes either through the
    /// meta-optimizer or when manually specified through the optimizers field.
    #[prost(message, optional, tag="5")]
    pub auto_parallel: ::std::option::Option<AutoParallelOptions>,
    /// If true, any optimization pass failing will cause the MetaOptimizer to
    /// stop with an error. By default - or when set to false, failing passes are
    /// skipped silently.
    #[prost(bool, tag="21")]
    pub fail_on_optimizer_errors: bool,
    #[prost(message, optional, tag="16")]
    pub scoped_allocator_opts: ::std::option::Option<ScopedAllocatorOptions>,
    /// If non-empty, will use this as an alternative way to specify a list of
    /// optimizations to turn on and the order of the optimizations (replacing the
    /// meta-optimizer).
    ///
    /// Of the RewriterConfig options, only the AutoParallel configuration options
    /// (the auto_parallel field) apply to manually requested optimization passes
    /// ("autoparallel"). Memory optimization passes ("memory") invoked here are
    /// not configurable (in contrast to memory optimization passes through the
    /// meta-optimizer) and act only on manual op annotations.
    ///
    /// Custom optimizers (see custom_optimizers) that are not part of this
    /// schedule will be run after - in the order that they were specified.
    #[prost(string, repeated, tag="100")]
    pub optimizers: ::std::vec::Vec<std::string::String>,
    /// list of CustomGraphOptimizers to apply.
    #[prost(message, repeated, tag="200")]
    pub custom_optimizers: ::std::vec::Vec<rewriter_config::CustomGraphOptimizer>,
    /// VerifierConfig specifying the verifiers to be run after every optimizer.
    #[prost(message, optional, tag="300")]
    pub inter_optimizer_verifier_config: ::std::option::Option<VerifierConfig>,
    /// VerifierConfig specifying the verifiers to be run at the end, after all
    /// optimizers have run.
    #[prost(message, optional, tag="301")]
    pub post_optimization_verifier_config: ::std::option::Option<VerifierConfig>,
}
pub mod rewriter_config {
    /// Message to describe custom graph optimizer and its parameters
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct CustomGraphOptimizer {
        #[prost(string, tag="1")]
        pub name: std::string::String,
        #[prost(map="string, message", tag="2")]
        pub parameter_map: ::std::collections::HashMap<std::string::String, super::AttrValue>,
    }
    // Configuration options for the meta-optimizer. Unless otherwise noted, these
    // configuration options do not apply to explicitly triggered optimization
    // passes in the optimizers field.

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Toggle {
        Default = 0,
        On = 1,
        Off = 2,
        /// Enable some aggressive optimizations that use assumptions that TF graphs
        /// may break. For example, assume the shape of a placeholder matches its
        /// actual feed.
        Aggressive = 3,
    }
    /// Enum controlling the number of times to run optimizers. The default is to
    /// run them twice.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum NumIterationsType {
        DefaultNumIters = 0,
        One = 1,
        Two = 2,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum MemOptType {
        /// The default setting (SCHEDULING and SWAPPING HEURISTICS only)
        DefaultMemOpt = 0,
        /// Disabled in the meta-optimizer.
        NoMemOpt = 1,
        /// Driven by manual op-level annotations.
        Manual = 2,
        // Driven by heuristics. The behavior of these heuristics is subject to
        // change. Currently includes an experimental recomputation and swapping
        // heuristics. Manual annotations are respected, but additional nodes are
        // selected automatically.

        /// Swapping heuristic will move a tensor from the GPU to the CPU and move
        /// it back when needed to reduce peak memory usage.
        SwappingHeuristics = 4,
        /// Recomputation heuristics will recompute ops (such as Relu activation)
        /// during backprop instead of storing them, reducing peak memory usage.
        RecomputationHeuristics = 5,
        /// Scheduling will split big ops such as AddN and try to enforce a schedule
        /// of the new computations that decreases peak memory usage.
        SchedulingHeuristics = 6,
        /// Use any combination of swapping and recomputation heuristics.
        Heuristics = 3,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GpuOptions {
    /// Fraction of the available GPU memory to allocate for each process.
    /// 1 means to allocate all of the GPU memory, 0.5 means the process
    /// allocates up to ~50% of the available GPU memory.
    ///
    /// GPU memory is pre-allocated unless the allow_growth option is enabled.
    ///
    /// If greater than 1.0, uses CUDA unified memory to potentially oversubscribe
    /// the amount of memory available on the GPU device by using host memory as a
    /// swap space. Accessing memory not available on the device will be
    /// significantly slower as that would require memory transfer between the host
    /// and the device. Options to reduce the memory requirement should be
    /// considered before enabling this option as this may come with a negative
    /// performance impact. Oversubscription using the unified memory requires
    /// Pascal class or newer GPUs and it is currently only supported on the Linux
    /// operating system. See
    /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
    /// for the detailed requirements.
    #[prost(double, tag="1")]
    pub per_process_gpu_memory_fraction: f64,
    /// If true, the allocator does not pre-allocate the entire specified
    /// GPU memory region, instead starting small and growing as needed.
    #[prost(bool, tag="4")]
    pub allow_growth: bool,
    /// The type of GPU allocation strategy to use.
    ///
    /// Allowed values:
    /// "": The empty string (default) uses a system-chosen default
    ///     which may change over time.
    ///
    /// "BFC": A "Best-fit with coalescing" algorithm, simplified from a
    ///        version of dlmalloc.
    #[prost(string, tag="2")]
    pub allocator_type: std::string::String,
    /// Delay deletion of up to this many bytes to reduce the number of
    /// interactions with gpu driver code.  If 0, the system chooses
    /// a reasonable default (several MBs).
    #[prost(int64, tag="3")]
    pub deferred_deletion_bytes: i64,
    /// A comma-separated list of GPU ids that determines the 'visible'
    /// to 'virtual' mapping of GPU devices.  For example, if TensorFlow
    /// can see 8 GPU devices in the process, and one wanted to map
    /// visible GPU devices 5 and 3 as "/device:GPU:0", and "/device:GPU:1",
    /// then one would specify this field as "5,3".  This field is similar in
    /// spirit to the CUDA_VISIBLE_DEVICES environment variable, except
    /// it applies to the visible GPU devices in the process.
    ///
    /// NOTE:
    /// 1. The GPU driver provides the process with the visible GPUs
    ///    in an order which is not guaranteed to have any correlation to
    ///    the *physical* GPU id in the machine.  This field is used for
    ///    remapping "visible" to "virtual", which means this operates only
    ///    after the process starts.  Users are required to use vendor
    ///    specific mechanisms (e.g., CUDA_VISIBLE_DEVICES) to control the
    ///    physical to visible device mapping prior to invoking TensorFlow.
    /// 2. In the code, the ids in this list are also called "platform GPU id"s,
    ///    and the 'virtual' ids of GPU devices (i.e. the ids in the device
    ///    name "/device:GPU:<id>") are also called "TF GPU id"s. Please
    ///    refer to third_party/tensorflow/core/common_runtime/gpu/gpu_id.h
    ///    for more information.
    #[prost(string, tag="5")]
    pub visible_device_list: std::string::String,
    /// In the event polling loop sleep this many microseconds between
    /// PollEvents calls, when the queue is not empty.  If value is not
    /// set or set to 0, gets set to a non-zero default.
    #[prost(int32, tag="6")]
    pub polling_active_delay_usecs: i32,
    /// This field is deprecated and ignored.
    #[prost(int32, tag="7")]
    pub polling_inactive_delay_msecs: i32,
    /// Force all tensors to be gpu_compatible. On a GPU-enabled TensorFlow,
    /// enabling this option forces all CPU tensors to be allocated with Cuda
    /// pinned memory. Normally, TensorFlow will infer which tensors should be
    /// allocated as the pinned memory. But in case where the inference is
    /// incomplete, this option can significantly speed up the cross-device memory
    /// copy performance as long as it fits the memory.
    /// Note that this option is not something that should be
    /// enabled by default for unknown or very large models, since all Cuda pinned
    /// memory is unpageable, having too much pinned memory might negatively impact
    /// the overall host system performance.
    #[prost(bool, tag="8")]
    pub force_gpu_compatible: bool,
    /// Everything inside experimental is subject to change and is not subject
    /// to API stability guarantees in
    /// https://www.tensorflow.org/guide/version_compat.
    #[prost(message, optional, tag="9")]
    pub experimental: ::std::option::Option<gpu_options::Experimental>,
}
pub mod gpu_options {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Experimental {
        /// The multi virtual device settings. If empty (not set), it will create
        /// single virtual device on each visible GPU, according to the settings
        /// in "visible_device_list" above. Otherwise, the number of elements in the
        /// list must be the same as the number of visible GPUs (after
        /// "visible_device_list" filtering if it is set), and the string represented
        /// device names (e.g. /device:GPU:<id>) will refer to the virtual
        /// devices and have the <id> field assigned sequentially starting from 0,
        /// according to the order they appear in this list and the "memory_limit"
        /// list inside each element. For example,
        ///   visible_device_list = "1,0"
        ///   virtual_devices { memory_limit: 1GB memory_limit: 2GB }
        ///   virtual_devices {}
        /// will create three virtual devices as:
        ///   /device:GPU:0 -> visible GPU 1 with 1GB memory
        ///   /device:GPU:1 -> visible GPU 1 with 2GB memory
        ///   /device:GPU:2 -> visible GPU 0 with all available memory
        ///
        /// NOTE:
        /// 1. It's invalid to set both this and "per_process_gpu_memory_fraction"
        ///    at the same time.
        /// 2. Currently this setting is per-process, not per-session. Using
        ///    different settings in different sessions within same process will
        ///    result in undefined behavior.
        #[prost(message, repeated, tag="1")]
        pub virtual_devices: ::std::vec::Vec<experimental::VirtualDevices>,
        /// If true, uses CUDA unified memory for memory allocations. If
        /// per_process_gpu_memory_fraction option is greater than 1.0, then unified
        /// memory is used regardless of the value for this field. See comments for
        /// per_process_gpu_memory_fraction field for more details and requirements
        /// of the unified memory. This option is useful to oversubscribe memory if
        /// multiple processes are sharing a single GPU while individually using less
        /// than 1.0 per process memory fraction.
        #[prost(bool, tag="2")]
        pub use_unified_memory: bool,
        /// If > 1, the number of device-to-device copy streams to create
        /// for each GPUDevice.  Default value is 0, which is automatically
        /// converted to 1.
        #[prost(int32, tag="3")]
        pub num_dev_to_dev_copy_streams: i32,
        /// If non-empty, defines a good GPU ring order on a single worker based on
        /// device interconnect.  This assumes that all workers have the same GPU
        /// topology.  Specify as a comma-separated string, e.g. "3,2,1,0,7,6,5,4".
        /// This ring order is used by the RingReducer implementation of
        /// CollectiveReduce, and serves as an override to automatic ring order
        /// generation in OrderTaskDeviceMap() during CollectiveParam resolution.
        #[prost(string, tag="4")]
        pub collective_ring_order: std::string::String,
        /// If true then extra work is done by GPUDevice and GPUBFCAllocator to
        /// keep track of when GPU memory is freed and when kernels actually
        /// complete so that we can know when a nominally free memory chunk
        /// is really not subject to pending use.
        #[prost(bool, tag="5")]
        pub timestamped_allocator: bool,
        // reserved id: 6

        /// Parameters for GPUKernelTracker.  By default no kernel tracking is done.
        /// Note that timestamped_allocator is only effective if some tracking is
        /// specified.
        ///
        /// If kernel_tracker_max_interval = n > 0, then a tracking event
        /// is inserted after every n kernels without an event.
        #[prost(int32, tag="7")]
        pub kernel_tracker_max_interval: i32,
        /// If kernel_tracker_max_bytes = n > 0, then a tracking event is
        /// inserted after every series of kernels allocating a sum of
        /// memory >= n.  If one kernel allocates b * n bytes, then one
        /// event will be inserted after it, but it will count as b against
        /// the pending limit.
        #[prost(int32, tag="8")]
        pub kernel_tracker_max_bytes: i32,
        /// If kernel_tracker_max_pending > 0 then no more than this many
        /// tracking events can be outstanding at a time.  An attempt to
        /// launch an additional kernel will stall until an event
        /// completes.
        #[prost(int32, tag="9")]
        pub kernel_tracker_max_pending: i32,
    }
    pub mod experimental {
        /// Configuration for breaking down a visible GPU into multiple "virtual"
        /// devices.
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct VirtualDevices {
            /// Per "virtual" device memory limit, in MB. The number of elements in
            /// the list is the number of virtual devices to create on the
            /// corresponding visible GPU (see "virtual_devices" below).
            /// If empty, it will create single virtual device taking all available
            /// memory from the device.
            ///
            /// For the concept of "visible" and "virtual" GPU, see the comments for
            /// "visible_device_list" above for more information.
            #[prost(float, repeated, tag="1")]
            pub memory_limit_mb: ::std::vec::Vec<f32>,
        }
    }
}
/// Options passed to the graph optimizer
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OptimizerOptions {
    /// If true, optimize the graph using common subexpression elimination.
    #[prost(bool, tag="1")]
    pub do_common_subexpression_elimination: bool,
    /// If true, perform constant folding optimization on the graph.
    #[prost(bool, tag="2")]
    pub do_constant_folding: bool,
    /// Constant folding optimization replaces tensors whose values can be
    /// predetermined, with constant nodes. To avoid inserting too large constants,
    /// the size of each constant created can be limited. If this value is zero, a
    /// default limit of 10 MiB will be applied. If constant folding optimization
    /// is disabled, this value is ignored.
    #[prost(int64, tag="6")]
    pub max_folded_constant_in_bytes: i64,
    /// If true, perform function inlining on the graph.
    #[prost(bool, tag="4")]
    pub do_function_inlining: bool,
    /// Overall optimization level. The actual optimizations applied will be the
    /// logical OR of the flags that this level implies and any flags already set.
    #[prost(enumeration="optimizer_options::Level", tag="3")]
    pub opt_level: i32,
    #[prost(enumeration="optimizer_options::GlobalJitLevel", tag="5")]
    pub global_jit_level: i32,
}
pub mod optimizer_options {
    /// Optimization level
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Level {
        /// L1 is the default level.
        /// Optimization performed at L1 :
        /// 1. Common subexpression elimination
        /// 2. Constant folding
        L1 = 0,
        /// No optimizations
        L0 = -1,
    }
    /// Control the use of the compiler/jit.  Experimental.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum GlobalJitLevel {
        /// Default setting ("off" now, but later expected to be "on")
        Default = 0,
        Off = -1,
        /// The following settings turn on compilation, with higher values being
        /// more aggressive.  Higher values may reduce opportunities for parallelism
        /// and may use more memory.  (At present, there is no distinction, but this
        /// is expected to change.)
        On1 = 1,
        On2 = 2,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphOptions {
    /// If true, use control flow to schedule the activation of Recv nodes.
    /// (Currently ignored.)
    #[prost(bool, tag="2")]
    pub enable_recv_scheduling: bool,
    /// Options controlling how graph is optimized.
    #[prost(message, optional, tag="3")]
    pub optimizer_options: ::std::option::Option<OptimizerOptions>,
    /// The number of steps to run before returning a cost model detailing
    /// the memory usage and performance of each node of the graph. 0 means
    /// no cost model.
    #[prost(int64, tag="4")]
    pub build_cost_model: i64,
    /// The number of steps to skip before collecting statistics for the
    /// cost model.
    #[prost(int64, tag="9")]
    pub build_cost_model_after: i64,
    /// Annotate each Node with Op output shape data, to the extent it can
    /// be statically inferred.
    #[prost(bool, tag="5")]
    pub infer_shapes: bool,
    /// Only place the subgraphs that are run, rather than the entire graph.
    ///
    /// This is useful for interactive graph building, where one might
    /// produce graphs that cannot be placed during the debugging
    /// process.  In particular, it allows the client to continue work in
    /// a session after adding a node to a graph whose placement
    /// constraints are unsatisfiable.
    #[prost(bool, tag="6")]
    pub place_pruned_graph: bool,
    /// If true, transfer float values between processes as bfloat16.
    #[prost(bool, tag="7")]
    pub enable_bfloat16_sendrecv: bool,
    /// If > 0, record a timeline every this many steps.
    /// EXPERIMENTAL: This currently has no effect in MasterSession.
    #[prost(int32, tag="8")]
    pub timeline_step: i32,
    /// Options that control the type and amount of graph rewriting.
    /// Not currently configurable via the public Python API (i.e. there is no API
    /// stability guarantee if you import RewriterConfig explicitly).
    #[prost(message, optional, tag="10")]
    pub rewrite_options: ::std::option::Option<RewriterConfig>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ThreadPoolOptionProto {
    /// The number of threads in the pool.
    ///
    /// 0 means the system picks a value based on where this option proto is used
    /// (see the declaration of the specific field for more info).
    #[prost(int32, tag="1")]
    pub num_threads: i32,
    /// The global name of the threadpool.
    ///
    /// If empty, then the threadpool is made and used according to the scope it's
    /// in - e.g., for a session threadpool, it is used by that session only.
    ///
    /// If non-empty, then:
    /// - a global threadpool associated with this name is looked
    ///   up or created. This allows, for example, sharing one threadpool across
    ///   many sessions (e.g., like the default behavior, if
    ///   inter_op_parallelism_threads is not configured), but still partitioning
    ///   into a large and small pool.
    /// - if the threadpool for this global_name already exists, then it is an
    ///   error if the existing pool was created using a different num_threads
    ///   value as is specified on this call.
    /// - threadpools created this way are never garbage collected.
    #[prost(string, tag="2")]
    pub global_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RpcOptions {
    /// If true, always use RPC to contact the session target.
    ///
    /// If false (the default option), TensorFlow may use an optimized
    /// transport for client-master communication that avoids the RPC
    /// stack. This option is primarily for used testing the RPC stack.
    #[prost(bool, tag="1")]
    pub use_rpc_for_inprocess_master: bool,
    /// The compression algorithm to be used. One of "deflate", "gzip".
    #[prost(string, tag="2")]
    pub compression_algorithm: std::string::String,
    /// If compression_algorithm is set, the compression level to be used.
    /// From 0 (no compression), up to 3.
    #[prost(int32, tag="3")]
    pub compression_level: i32,
    /// Setting cache_rpc_response to true will enable sender side caching of
    /// response for RecvTensorAsync and RecvBufAsync to allow receiver to retry
    /// requests . This is only necessary when the network fabric is experiencing a
    /// significant error rate.  Without it we'll fail a step on an network error,
    /// while with it we'll be able to complete long steps (like complex
    /// initializations) in the face of some network errors during RecvTensor.
    #[prost(bool, tag="4")]
    pub cache_rpc_response: bool,
    /// Disables TCP connection sharing when opening a new RPC channel.
    #[prost(bool, tag="5")]
    pub disable_session_connection_sharing: bool,
}
/// Metadata about the session.
///
/// This can be used by the runtime and the Ops for debugging, monitoring, etc.
///
/// The (name, version) tuple is expected to be a unique identifier for
/// sessions within the same process.
///
/// NOTE: This is currently used and propagated only by the direct session.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SessionMetadata {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The version is optional. If set, needs to be >= 0.
    #[prost(int64, tag="2")]
    pub version: i64,
}
/// Session configuration parameters.
/// The system picks appropriate values for fields that are not set.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConfigProto {
    /// Map from device type name (e.g., "CPU" or "GPU" ) to maximum
    /// number of devices of that type to use.  If a particular device
    /// type is not found in the map, the system picks an appropriate
    /// number.
    #[prost(map="string, int32", tag="1")]
    pub device_count: ::std::collections::HashMap<std::string::String, i32>,
    /// The execution of an individual op (for some op types) can be
    /// parallelized on a pool of intra_op_parallelism_threads.
    /// 0 means the system picks an appropriate number.
    #[prost(int32, tag="2")]
    pub intra_op_parallelism_threads: i32,
    /// Nodes that perform blocking operations are enqueued on a pool of
    /// inter_op_parallelism_threads available in each process.
    ///
    /// 0 means the system picks an appropriate number.
    /// Negative means all operations are performed in caller's thread.
    ///
    /// Note that the first Session created in the process sets the
    /// number of threads for all future sessions unless use_per_session_threads is
    /// true or session_inter_op_thread_pool is configured.
    #[prost(int32, tag="5")]
    pub inter_op_parallelism_threads: i32,
    /// If true, use a new set of threads for this session rather than the global
    /// pool of threads. Only supported by direct sessions.
    ///
    /// If false, use the global threads created by the first session, or the
    /// per-session thread pools configured by session_inter_op_thread_pool.
    ///
    /// This option is deprecated. The same effect can be achieved by setting
    /// session_inter_op_thread_pool to have one element, whose num_threads equals
    /// inter_op_parallelism_threads.
    #[prost(bool, tag="9")]
    pub use_per_session_threads: bool,
    /// This option is experimental - it may be replaced with a different mechanism
    /// in the future.
    ///
    /// Configures session thread pools. If this is configured, then RunOptions for
    /// a Run call can select the thread pool to use.
    ///
    /// The intended use is for when some session invocations need to run in a
    /// background pool limited to a small number of threads:
    /// - For example, a session may be configured to have one large pool (for
    /// regular compute) and one small pool (for periodic, low priority work);
    /// using the small pool is currently the mechanism for limiting the inter-op
    /// parallelism of the low priority work.  Note that it does not limit the
    /// parallelism of work spawned by a single op kernel implementation.
    /// - Using this setting is normally not needed in training, but may help some
    /// serving use cases.
    /// - It is also generally recommended to set the global_name field of this
    /// proto, to avoid creating multiple large pools. It is typically better to
    /// run the non-low-priority work, even across sessions, in a single large
    /// pool.
    #[prost(message, repeated, tag="12")]
    pub session_inter_op_thread_pool: ::std::vec::Vec<ThreadPoolOptionProto>,
    /// Assignment of Nodes to Devices is recomputed every placement_period
    /// steps until the system warms up (at which point the recomputation
    /// typically slows down automatically).
    #[prost(int32, tag="3")]
    pub placement_period: i32,
    /// When any filters are present sessions will ignore all devices which do not
    /// match the filters. Each filter can be partially specified, e.g. "/job:ps"
    /// "/job:worker/replica:3", etc.
    #[prost(string, repeated, tag="4")]
    pub device_filters: ::std::vec::Vec<std::string::String>,
    /// Options that apply to all GPUs.
    #[prost(message, optional, tag="6")]
    pub gpu_options: ::std::option::Option<GpuOptions>,
    /// Whether soft placement is allowed. If allow_soft_placement is true,
    /// an op will be placed on CPU if
    ///   1. there's no GPU implementation for the OP
    /// or
    ///   2. no GPU devices are known or registered
    /// or
    ///   3. need to co-locate with reftype input(s) which are from CPU.
    #[prost(bool, tag="7")]
    pub allow_soft_placement: bool,
    /// Whether device placements should be logged.
    #[prost(bool, tag="8")]
    pub log_device_placement: bool,
    /// Options that apply to all graphs.
    #[prost(message, optional, tag="10")]
    pub graph_options: ::std::option::Option<GraphOptions>,
    /// Global timeout for all blocking operations in this session.  If non-zero,
    /// and not overridden on a per-operation basis, this value will be used as the
    /// deadline for all blocking operations.
    #[prost(int64, tag="11")]
    pub operation_timeout_in_ms: i64,
    /// Options that apply when this session uses the distributed runtime.
    #[prost(message, optional, tag="13")]
    pub rpc_options: ::std::option::Option<RpcOptions>,
    /// Optional list of all workers to use in this session.
    #[prost(message, optional, tag="14")]
    pub cluster_def: ::std::option::Option<ClusterDef>,
    /// If true, any resources such as Variables used in the session will not be
    /// shared with other sessions. However, when clusterspec propagation is
    /// enabled, this field is ignored and sessions are always isolated.
    #[prost(bool, tag="15")]
    pub isolate_session_state: bool,
    #[prost(message, optional, tag="16")]
    pub experimental: ::std::option::Option<config_proto::Experimental>,
}
pub mod config_proto {
    /// Everything inside Experimental is subject to change and is not subject
    /// to API stability guarantees in
    /// https://www.tensorflow.org/guide/version_compat.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Experimental {
        /// Task name for group resolution.
        #[prost(string, tag="1")]
        pub collective_group_leader: std::string::String,
        /// Which executor to use, the default executor will be used
        /// if it is an empty string or "DEFAULT"
        #[prost(string, tag="3")]
        pub executor_type: std::string::String,
        /// Guidance to formatting of large RecvBuf fields for transfer.
        /// Any positive value sets the max chunk size.  0 defaults to 4096.
        /// Any negative value indicates no max, i.e. one chunk only.
        #[prost(int32, tag="4")]
        pub recv_buf_max_chunk: i32,
        /// If true, and supported by the platform, the runtime will attempt to
        /// use NUMA affinity where applicable.  One consequence will be the
        /// existence of as many CPU devices as there are available NUMA nodes.
        #[prost(bool, tag="5")]
        pub use_numa_affinity: bool,
        /// If true, make collective op execution order sequential and deterministic
        /// for potentially concurrent collective instances.
        #[prost(bool, tag="6")]
        pub collective_deterministic_sequential_execution: bool,
        /// If true, use NCCL for CollectiveOps.  This feature is highly
        /// experimental.
        #[prost(bool, tag="7")]
        pub collective_nccl: bool,
        /// In the following, session state means the value of a variable, elements
        /// in a hash table, or any other resource, accessible by worker sessions
        /// held by a TF server.
        ///
        /// When ClusterSpec propagation is enabled, the value of
        /// isolate_session_state is ignored when deciding whether to share session
        /// states in a TF server (for backwards compatibility reasons).
        /// - If share_session_state_in_clusterspec_propagation is true, the session
        /// states are shared.
        /// - If share_session_state_in_clusterspec_propagation is false, session
        /// states are isolated.
        ///
        /// When clusterspec propagation is not used, the value of
        /// share_session_state_in_clusterspec_propagation is ignored when deciding
        /// whether to share session states in a TF server.
        /// - If isolate_session_state is true, session states are isolated.
        /// - If isolate_session_state is false, session states are shared.
        ///
        /// TODO(b/129330037): Add a single API that consistently treats
        /// isolate_session_state and ClusterSpec propagation.
        #[prost(bool, tag="8")]
        pub share_session_state_in_clusterspec_propagation: bool,
        /// If using a direct session, disable spinning while waiting for work in
        /// the thread pool. This may result in higher latency for completing ops,
        /// but in the case where there is a lot of spinning may result in lower
        /// CPU usage.
        #[prost(bool, tag="9")]
        pub disable_thread_spinning: bool,
        /// When true, WorkerSessions are created with device attributes from the
        /// full cluster.
        /// This is helpful when a worker wants to partition a graph
        /// (for example during a PartitionedCallOp).
        #[prost(bool, tag="10")]
        pub share_cluster_devices_in_session: bool,
        /// Metadata about the session.
        ///
        /// If set, this can be used by the runtime and the Ops for debugging,
        /// monitoring, etc.
        ///
        /// NOTE: This is currently used and propagated only by the direct session.
        #[prost(message, optional, tag="11")]
        pub session_metadata: ::std::option::Option<super::SessionMetadata>,
        /// If true, the session may treat the graph as being static for optimization
        /// purposes.
        ///
        /// If this option is set to true when a session is created, the full
        /// GraphDef must be passed in a single call to Session::Create(), and
        /// Session::Extend() may not be supported.
        #[prost(bool, tag="12")]
        pub optimize_for_static_graph: bool,
    }
}
/// Options for a single Run() call.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunOptions {
    #[prost(enumeration="run_options::TraceLevel", tag="1")]
    pub trace_level: i32,
    /// Time to wait for operation to complete in milliseconds.
    #[prost(int64, tag="2")]
    pub timeout_in_ms: i64,
    /// The thread pool to use, if session_inter_op_thread_pool is configured.
    /// To use the caller thread set this to -1 - this uses the caller thread
    /// to execute Session::Run() and thus avoids a context switch. Using the
    /// caller thread to execute Session::Run() should be done ONLY for simple
    /// graphs, where the overhead of an additional context switch is
    /// comparable with the overhead of Session::Run().
    #[prost(int32, tag="3")]
    pub inter_op_thread_pool: i32,
    /// Whether the partition graph(s) executed by the executor(s) should be
    /// outputted via RunMetadata.
    #[prost(bool, tag="5")]
    pub output_partition_graphs: bool,
    /// EXPERIMENTAL.  Options used to initialize DebuggerState, if enabled.
    #[prost(message, optional, tag="6")]
    pub debug_options: ::std::option::Option<DebugOptions>,
    /// When enabled, causes tensor allocation information to be included in
    /// the error message when the Run() call fails because the allocator ran
    /// out of memory (OOM).
    ///
    /// Enabling this option can slow down the Run() call.
    #[prost(bool, tag="7")]
    pub report_tensor_allocations_upon_oom: bool,
    #[prost(message, optional, tag="8")]
    pub experimental: ::std::option::Option<run_options::Experimental>,
}
pub mod run_options {
    /// Everything inside Experimental is subject to change and is not subject
    /// to API stability guarantees in
    /// https://www.tensorflow.org/guide/version_compat.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Experimental {
        /// If non-zero, declares that this graph is going to use collective
        /// ops and must synchronize step_ids with any other graph with this
        /// same group_key value (in a distributed computation where tasks
        /// run disjoint graphs).
        #[prost(int64, tag="1")]
        pub collective_graph_key: i64,
        /// If true, then operations (using the inter-op pool) across all
        /// session::run() calls will be centrally scheduled, optimizing for (median
        /// and tail) latency.
        /// Consider using this option for CPU-bound workloads like inference.
        #[prost(bool, tag="2")]
        pub use_run_handler_pool: bool,
    }
    /// TODO(pbar) Turn this into a TraceOptions proto which allows
    /// tracing to be controlled in a more orthogonal manner?
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum TraceLevel {
        NoTrace = 0,
        SoftwareTrace = 1,
        HardwareTrace = 2,
        FullTrace = 3,
    }
}
/// Metadata output (i.e., non-Tensor) for a single Run() call.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunMetadata {
    /// Statistics traced for this step. Populated if tracing is turned on via the
    /// "RunOptions" proto.
    /// EXPERIMENTAL: The format and set of events may change in future versions.
    #[prost(message, optional, tag="1")]
    pub step_stats: ::std::option::Option<StepStats>,
    /// The cost graph for the computation defined by the run call.
    #[prost(message, optional, tag="2")]
    pub cost_graph: ::std::option::Option<CostGraphDef>,
    /// Graphs of the partitions executed by executors.
    #[prost(message, repeated, tag="3")]
    pub partition_graphs: ::std::vec::Vec<GraphDef>,
    /// This is only populated for graphs that are run as functions in TensorFlow
    /// V2. There will be an entry below for each function that is traced.
    /// The main use cases of the post_optimization_graph and the partition_graphs
    /// is to give the caller insight into the graphs that were actually run by the
    /// runtime. Additional information (such as those in step_stats) will match
    /// these graphs.
    /// We also include the pre_optimization_graph since it is usually easier to
    /// read, and is helpful in situations where the caller wants to get a high
    /// level idea of what the built graph looks like (since the various graph
    /// optimization passes might change the structure of the graph significantly).
    #[prost(message, repeated, tag="4")]
    pub function_graphs: ::std::vec::Vec<run_metadata::FunctionGraphs>,
}
pub mod run_metadata {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FunctionGraphs {
        /// TODO(nareshmodi): Include some sort of function/cache-key identifier?
        #[prost(message, repeated, tag="1")]
        pub partition_graphs: ::std::vec::Vec<super::GraphDef>,
        #[prost(message, optional, tag="2")]
        pub pre_optimization_graph: ::std::option::Option<super::GraphDef>,
        #[prost(message, optional, tag="3")]
        pub post_optimization_graph: ::std::option::Option<super::GraphDef>,
    }
}
/// Defines a connection between two tensors in a `GraphDef`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorConnection {
    /// A tensor name. The value of this tensor will be substituted for
    /// the tensor named in `to_tensor`.
    #[prost(string, tag="1")]
    pub from_tensor: std::string::String,
    /// A tensor name. The value of this tensor will be bound to the
    /// value of the tensor named in `from_tensor`.
    #[prost(string, tag="2")]
    pub to_tensor: std::string::String,
}
/// Defines a subgraph in another `GraphDef` as a set of feed points and nodes
/// to be fetched or executed.
///
/// Compare with the arguments to `Session::Run()`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CallableOptions {
    /// Tensors to be fed in the callable. Each feed is the name of a tensor.
    #[prost(string, repeated, tag="1")]
    pub feed: ::std::vec::Vec<std::string::String>,
    /// Fetches. A list of tensor names. The caller of the callable expects a
    /// tensor to be returned for each fetch[i] (see RunStepResponse.tensor). The
    /// order of specified fetches does not change the execution order.
    #[prost(string, repeated, tag="2")]
    pub fetch: ::std::vec::Vec<std::string::String>,
    /// Target Nodes. A list of node names. The named nodes will be run by the
    /// callable but their outputs will not be returned.
    #[prost(string, repeated, tag="3")]
    pub target: ::std::vec::Vec<std::string::String>,
    /// Options that will be applied to each run.
    #[prost(message, optional, tag="4")]
    pub run_options: ::std::option::Option<RunOptions>,
    /// Tensors to be connected in the callable. Each TensorConnection denotes
    /// a pair of tensors in the graph, between which an edge will be created
    /// in the callable.
    #[prost(message, repeated, tag="5")]
    pub tensor_connection: ::std::vec::Vec<TensorConnection>,
    /// The Tensor objects fed in the callable and fetched from the callable
    /// are expected to be backed by host (CPU) memory by default.
    ///
    /// The options below allow changing that - feeding tensors backed by
    /// device memory, or returning tensors that are backed by device memory.
    ///
    /// The maps below map the name of a feed/fetch tensor (which appears in
    /// 'feed' or 'fetch' fields above), to the fully qualified name of the device
    /// owning the memory backing the contents of the tensor.
    ///
    /// For example, creating a callable with the following options:
    ///
    /// CallableOptions {
    ///   feed: "a:0"
    ///   feed: "b:0"
    ///
    ///   fetch: "x:0"
    ///   fetch: "y:0"
    ///
    ///   feed_devices: {
    ///     "a:0": "/job:localhost/replica:0/task:0/device:GPU:0"
    ///   }
    ///
    ///   fetch_devices: {
    ///     "y:0": "/job:localhost/replica:0/task:0/device:GPU:0"
    ///  }
    /// }
    ///
    /// means that the Callable expects:
    /// - The first argument ("a:0") is a Tensor backed by GPU memory.
    /// - The second argument ("b:0") is a Tensor backed by host memory.
    /// and of its return values:
    /// - The first output ("x:0") will be backed by host memory.
    /// - The second output ("y:0") will be backed by GPU memory.
    ///
    /// FEEDS:
    /// It is the responsibility of the caller to ensure that the memory of the fed
    /// tensors will be correctly initialized and synchronized before it is
    /// accessed by operations executed during the call to Session::RunCallable().
    ///
    /// This is typically ensured by using the TensorFlow memory allocators
    /// (Device::GetAllocator()) to create the Tensor to be fed.
    ///
    /// Alternatively, for CUDA-enabled GPU devices, this typically means that the
    /// operation that produced the contents of the tensor has completed, i.e., the
    /// CUDA stream has been synchronized (e.g., via cuCtxSynchronize() or
    /// cuStreamSynchronize()).
    #[prost(map="string, string", tag="6")]
    pub feed_devices: ::std::collections::HashMap<std::string::String, std::string::String>,
    #[prost(map="string, string", tag="7")]
    pub fetch_devices: ::std::collections::HashMap<std::string::String, std::string::String>,
    /// By default, RunCallable() will synchronize the GPU stream before returning
    /// fetched tensors on a GPU device, to ensure that the values in those tensors
    /// have been produced. This simplifies interacting with the tensors, but
    /// potentially incurs a performance hit.
    ///
    /// If this options is set to true, the caller is responsible for ensuring
    /// that the values in the fetched tensors have been produced before they are
    /// used. The caller can do this by invoking `Device::Sync()` on the underlying
    /// device(s), or by feeding the tensors back to the same Session using
    /// `feed_devices` with the same corresponding device name.
    #[prost(bool, tag="8")]
    pub fetch_skip_sync: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileOptions {
    /// We don't collect the dataset ops by default for better trace-viewer
    /// scalability. The caller can mannually set this field to include the ops.
    #[prost(bool, tag="1")]
    pub include_dataset_ops: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolRequestOptions {
    /// Required formats for the tool, it should be one of "json", "proto", "raw"
    /// etc. If not specified (backward compatible), use default format, i.e. most
    /// tools use json format.
    #[prost(string, tag="2")]
    pub output_formats: std::string::String,
    /// Whether save the result directly to repository or pass it back to caller.
    /// Default to false for backward compatibilities.
    #[prost(bool, tag="3")]
    pub save_to_repo: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileRequest {
    /// In future, the caller will be able to customize when profiling starts and
    /// stops. For now, it collects `duration_ms` milliseconds worth of data.
    #[prost(uint64, tag="1")]
    pub duration_ms: u64,
    /// The maximum number of events to return. By default (value 0), return all
    /// events.
    #[prost(uint64, tag="2")]
    pub max_events: u64,
    /// Required profiling tools name such as "input_pipeline_analyzer" etc
    #[prost(string, repeated, tag="3")]
    pub tools: ::std::vec::Vec<std::string::String>,
    /// Specifies the requirement for each tools.
    #[prost(map="string, message", tag="8")]
    pub tool_options: ::std::collections::HashMap<std::string::String, ToolRequestOptions>,
    /// Optional profiling options that control how a TF session will be profiled.
    #[prost(message, optional, tag="4")]
    pub opts: ::std::option::Option<ProfileOptions>,
    /// The place where we will dump profile data. We will normally use
    /// MODEL_DIR/plugin/profile/ as our repository root.
    #[prost(string, tag="5")]
    pub repository_root: std::string::String,
    /// The user provided profile session identifier.
    #[prost(string, tag="6")]
    pub session_id: std::string::String,
    /// The hostname of system where the profile should happen.
    /// We use it as identifier in part of our output filename.
    #[prost(string, tag="7")]
    pub host_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileToolData {
    /// The file name which this data is associated (e.g. "input_pipeline.json",
    /// "cluster_xxx.memory_viewer.json").
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The data payload (likely json) for the specific tool.
    #[prost(bytes, tag="2")]
    pub data: std::vec::Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileResponse {
    /// Graphs of programs executed on devices during the profiling period.
    #[prost(message, repeated, tag="2")]
    pub computation_graph: ::std::vec::Vec<GraphDef>,
    /// Performance profile that can be used to annotate HLO operations in the
    /// computation graph.
    #[prost(message, optional, tag="5")]
    pub hlo_metadata: ::std::option::Option<RunMetadata>,
    /// Encoded Trace proto message that contains metadata about the trace captured
    /// during the profiling period. Describes the devices and resources that
    /// 'trace_events' refers to.
    #[prost(bytes, tag="3")]
    pub encoded_trace: std::vec::Vec<u8>,
    /// Assembles a hierarchical performance profile based on HLOs in trace events.
    /// If the trace covers multiple programs, the longest-running one is analyzed.
    /// See op_profile.proto for the detailed semantics of the returned profile.
    #[prost(message, optional, tag="4")]
    pub op_profile: ::std::option::Option<profiler::op_profile::Profile>,
    /// Data payload for each required tools.
    #[prost(message, repeated, tag="6")]
    pub tool_data: ::std::vec::Vec<ProfileToolData>,
    /// When we write profiling data directly to repository directory, we need a
    /// way to figure out whether the captured trace is empty (due to idle TPU).
    #[prost(bool, tag="7")]
    pub empty_trace: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MonitorRequest {
    /// Duration for which to profile between each update.
    #[prost(uint64, tag="1")]
    pub duration_ms: u64,
    /// Indicates the level at which we want to monitor. Currently, two levels are
    /// supported:
    /// Level 1: An ultra lightweight mode that captures only some utilization
    /// metrics.
    /// Level 2: More verbose than level 1. Collects utilization metrics, device
    /// information, step time information, etc. Do not use this option if the TPU
    /// host is being very heavily used.
    #[prost(int32, tag="2")]
    pub monitoring_level: i32,
    /// True to display timestamp in monitoring result.
    #[prost(bool, tag="3")]
    pub timestamp: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MonitorResponse {
    /// Properly formatted string data that can be directly returned back to user.
    #[prost(string, tag="1")]
    pub data: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NewProfileSessionRequest {
    #[prost(message, optional, tag="1")]
    pub request: ::std::option::Option<ProfileRequest>,
    #[prost(string, tag="2")]
    pub repository_root: std::string::String,
    #[prost(string, repeated, tag="3")]
    pub hosts: ::std::vec::Vec<std::string::String>,
    #[prost(string, tag="4")]
    pub session_id: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NewProfileSessionResponse {
    /// Auxiliary error_message.
    #[prost(string, tag="1")]
    pub error_message: std::string::String,
    /// Whether all hosts had returned a empty trace.
    #[prost(bool, tag="2")]
    pub empty_trace: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EnumProfileSessionsAndToolsRequest {
    #[prost(string, tag="1")]
    pub repository_root: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileSessionInfo {
    #[prost(string, tag="1")]
    pub session_id: std::string::String,
    /// Which tool data is available for consumption.
    #[prost(string, repeated, tag="2")]
    pub available_tools: ::std::vec::Vec<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EnumProfileSessionsAndToolsResponse {
    /// Auxiliary error_message.
    #[prost(string, tag="1")]
    pub error_message: std::string::String,
    /// If success, the returned sessions information are stored here.
    #[prost(message, repeated, tag="2")]
    pub sessions: ::std::vec::Vec<ProfileSessionInfo>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileSessionDataRequest {
    #[prost(string, tag="1")]
    pub repository_root: std::string::String,
    #[prost(string, tag="2")]
    pub session_id: std::string::String,
    /// Which host the data is associated. if empty, data from all hosts are
    /// aggregated.
    #[prost(string, tag="5")]
    pub host_name: std::string::String,
    /// Which tool
    #[prost(string, tag="3")]
    pub tool_name: std::string::String,
    /// Tool's specific parameters. e.g. TraceViewer's viewport etc
    #[prost(map="string, string", tag="4")]
    pub parameters: ::std::collections::HashMap<std::string::String, std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileSessionDataResponse {
    /// Auxiliary error_message.
    #[prost(string, tag="1")]
    pub error_message: std::string::String,
    /// Output format. e.g. "json" or "proto" or "blob"
    #[prost(string, tag="2")]
    pub output_format: std::string::String,
    /// TODO(jiesun): figure out whether to put bytes or oneof tool specific proto.
    #[prost(bytes, tag="3")]
    pub output: std::vec::Vec<u8>,
}
/// Protocol buffer representing a QueueRunner.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct QueueRunnerDef {
    /// Queue name.
    #[prost(string, tag="1")]
    pub queue_name: std::string::String,
    /// A list of enqueue operations.
    #[prost(string, repeated, tag="2")]
    pub enqueue_op_name: ::std::vec::Vec<std::string::String>,
    /// The operation to run to close the queue.
    #[prost(string, tag="3")]
    pub close_op_name: std::string::String,
    /// The operation to run to cancel the queue.
    #[prost(string, tag="4")]
    pub cancel_op_name: std::string::String,
    /// A list of exception types considered to signal a safely closed queue
    /// if raised during enqueue operations.
    #[prost(enumeration="error::Code", repeated, tag="5")]
    pub queue_closed_exception_types: ::std::vec::Vec<i32>,
}
/// Protocol buffer representing the configuration of a Saver.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SaverDef {
    /// The name of the tensor in which to specify the filename when saving or
    /// restoring a model checkpoint.
    #[prost(string, tag="1")]
    pub filename_tensor_name: std::string::String,
    /// The operation to run when saving a model checkpoint.
    #[prost(string, tag="2")]
    pub save_tensor_name: std::string::String,
    /// The operation to run when restoring a model checkpoint.
    #[prost(string, tag="3")]
    pub restore_op_name: std::string::String,
    /// Maximum number of checkpoints to keep.  If 0, no checkpoints are deleted.
    #[prost(int32, tag="4")]
    pub max_to_keep: i32,
    /// Shard the save files, one per device that has Variable nodes.
    #[prost(bool, tag="5")]
    pub sharded: bool,
    /// How often to keep an additional checkpoint. If not specified, only the last
    /// "max_to_keep" checkpoints are kept; if specified, in addition to keeping
    /// the last "max_to_keep" checkpoints, an additional checkpoint will be kept
    /// for every n hours of training.
    #[prost(float, tag="6")]
    pub keep_checkpoint_every_n_hours: f32,
    #[prost(enumeration="saver_def::CheckpointFormatVersion", tag="7")]
    pub version: i32,
}
pub mod saver_def {
    /// A version number that identifies a different on-disk checkpoint format.
    /// Usually, each subclass of BaseSaverBuilder works with a particular
    /// version/format.  However, it is possible that the same builder may be
    /// upgraded to support a newer checkpoint format in the future.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum CheckpointFormatVersion {
        /// Internal legacy format.
        Legacy = 0,
        /// Deprecated format: tf.Saver() which works with tensorflow::table::Table.
        V1 = 1,
        /// Current format: more efficient.
        V2 = 2,
    }
}
/// Protocol buffer representing a CriticalSection.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CriticalSectionDef {
    /// Name of the critical section handle.
    #[prost(string, tag="1")]
    pub critical_section_name: std::string::String,
}
/// Protocol buffer representing a CriticalSection execution.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CriticalSectionExecutionDef {
    /// Name of the critical section handle.
    #[prost(string, tag="1")]
    pub execute_in_critical_section_name: std::string::String,
    /// Whether this operation requires exclusive access to its resources,
    /// (i.e., no other CriticalSections may request the same resources).
    #[prost(bool, tag="2")]
    pub exclusive_resource_access: bool,
}
/// Extra data needed on a non-RDMA RecvBufResponse.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecvBufRespExtra {
    #[prost(bytes, repeated, tag="1")]
    pub tensor_content: ::std::vec::Vec<std::vec::Vec<u8>>,
}
// Protos used in the tensor bundle module (tf/core/util/tensor_bundle/).

/// Special header that is associated with a bundle.
///
/// TODO(zongheng,zhifengc): maybe in the future, we can add information about
/// which binary produced this checkpoint, timestamp, etc. Sometime, these can be
/// valuable debugging information. And if needed, these can be used as defensive
/// information ensuring reader (binary version) of the checkpoint and the writer
/// (binary version) must match within certain range, etc.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BundleHeaderProto {
    /// Number of data files in the bundle.
    #[prost(int32, tag="1")]
    pub num_shards: i32,
    #[prost(enumeration="bundle_header_proto::Endianness", tag="2")]
    pub endianness: i32,
    /// Versioning of the tensor bundle format.
    #[prost(message, optional, tag="3")]
    pub version: ::std::option::Option<VersionDef>,
}
pub mod bundle_header_proto {
    /// An enum indicating the endianness of the platform that produced this
    /// bundle.  A bundle can only be read by a platform with matching endianness.
    /// Defaults to LITTLE, as most modern platforms are little-endian.
    ///
    /// Affects the binary tensor data bytes only, not the metadata in protobufs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Endianness {
        Little = 0,
        Big = 1,
    }
}
/// Describes the metadata related to a checkpointed tensor.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BundleEntryProto {
    /// The tensor dtype and shape.
    #[prost(enumeration="DataType", tag="1")]
    pub dtype: i32,
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    /// The binary content of the tensor lies in:
    ///   File "shard_id": bytes [offset, offset + size).
    #[prost(int32, tag="3")]
    pub shard_id: i32,
    #[prost(int64, tag="4")]
    pub offset: i64,
    #[prost(int64, tag="5")]
    pub size: i64,
    /// The CRC32C checksum of the tensor bytes.
    #[prost(fixed32, tag="6")]
    pub crc32c: u32,
    /// Iff present, this entry represents a partitioned tensor.  The previous
    /// fields are interpreted as follows:
    ///
    ///   "dtype", "shape": describe the full tensor.
    ///   "shard_id", "offset", "size", "crc32c": all IGNORED.
    ///      These information for each slice can be looked up in their own
    ///      BundleEntryProto, keyed by each "slice_name".
    #[prost(message, repeated, tag="7")]
    pub slices: ::std::vec::Vec<TensorSliceProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InterconnectLink {
    #[prost(int32, tag="1")]
    pub device_id: i32,
    #[prost(string, tag="2")]
    pub r#type: std::string::String,
    #[prost(int32, tag="3")]
    pub strength: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LocalLinks {
    #[prost(message, repeated, tag="1")]
    pub link: ::std::vec::Vec<InterconnectLink>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceLocality {
    /// Optional bus locality of device.  Default value of 0 means
    /// no specific locality.  Specific localities are indexed from 1.
    #[prost(int32, tag="1")]
    pub bus_id: i32,
    /// Optional NUMA locality of device.
    #[prost(int32, tag="2")]
    pub numa_node: i32,
    /// Optional local interconnect links to other devices.
    #[prost(message, optional, tag="3")]
    pub links: ::std::option::Option<LocalLinks>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceAttributes {
    /// Fully specified name of the device within a cluster.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// String representation of device_type.
    #[prost(string, tag="2")]
    pub device_type: std::string::String,
    /// Memory capacity of device in bytes.
    #[prost(int64, tag="4")]
    pub memory_limit: i64,
    /// Platform-specific data about device that may be useful
    /// for supporting efficient data transfers.
    #[prost(message, optional, tag="5")]
    pub locality: ::std::option::Option<DeviceLocality>,
    /// A device is assigned a global unique number each time it is
    /// initialized. "incarnation" should never be 0.
    #[prost(fixed64, tag="6")]
    pub incarnation: u64,
    /// String representation of the physical device that this device maps to.
    #[prost(string, tag="7")]
    pub physical_device_desc: std::string::String,
}
/// A pair of tensor name and tensor values.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NamedTensorProto {
    /// Name of the tensor.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The client can populate a TensorProto using a tensorflow::Tensor`, or
    /// directly using the protobuf field accessors.
    ///
    /// The client specifies whether the returned tensor values should be
    /// filled tensor fields (float_val, int_val, etc.) or encoded in a
    /// compact form in tensor.tensor_content.
    #[prost(message, optional, tag="2")]
    pub tensor: ::std::option::Option<TensorProto>,
}
////////////////////////////////////////////////////////////////////////////////
//
// CreateSession method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateSessionRequest {
    /// The initial graph definition.
    #[prost(message, optional, tag="1")]
    pub graph_def: ::std::option::Option<GraphDef>,
    /// Configuration options.
    #[prost(message, optional, tag="2")]
    pub config: ::std::option::Option<ConfigProto>,
    /// The target string used from the client's perspective.
    #[prost(string, tag="3")]
    pub target: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateSessionResponse {
    /// The session handle to be used in subsequent calls for the created session.
    ///
    /// The client must arrange to call CloseSession with this returned
    /// session handle to close the session.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// The initial version number for the graph, to be used in the next call
    /// to ExtendSession.
    #[prost(int64, tag="2")]
    pub graph_version: i64,
}
////////////////////////////////////////////////////////////////////////////////
//
// ExtendSession method request/response protos.
//
// The "graph_def" specifies a set of nodes to be added to the session's graph.
//
// A typical "graph_def" will contain:
//
// * Zero or more new nodes with names that do not exist in the server-side
//   graph. These will be added to the graph.
//
// PRECONDITION: The server-side current version is req.current_version.
//   None of the names in req.graph_def appeared in previous successful calls to
//   CreateSession or ExtendSession with the same session_handle.
// POSTCONDITION: The server-side current version is resp.new_version.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExtendSessionRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// REQUIRED: The nodes to be added to the session's graph. If any node has
    /// the same name as an existing node, the operation will fail with
    /// ILLEGAL_ARGUMENT.
    #[prost(message, optional, tag="2")]
    pub graph_def: ::std::option::Option<GraphDef>,
    /// REQUIRED: The version number of the graph to be extended. This will be
    /// tested against the current server-side version number, and the operation
    /// will fail with FAILED_PRECONDITION if they do not match.
    #[prost(int64, tag="3")]
    pub current_graph_version: i64,
}
/// TODO(mrry): Return something about the operation?
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExtendSessionResponse {
    /// The new version number for the extended graph, to be used in the next call
    /// to ExtendSession.
    #[prost(int64, tag="4")]
    pub new_graph_version: i64,
}
////////////////////////////////////////////////////////////////////////////////
//
// RunStep method request/response protos.
//
// The caller should provide the feeds needed by the graph and specify
// what nodes should be fetched.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunStepRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// Tensors to be fed in the step. Each feed is a named tensor.
    #[prost(message, repeated, tag="2")]
    pub feed: ::std::vec::Vec<NamedTensorProto>,
    /// Fetches. A list of tensor names. The caller expects a tensor to
    /// be returned for each fetch[i] (see RunStepResponse.tensor). The
    /// order of specified fetches does not change the execution order.
    #[prost(string, repeated, tag="3")]
    pub fetch: ::std::vec::Vec<std::string::String>,
    /// Target Nodes. A list of node names. The named nodes will be run
    /// to but their outputs will not be fetched.
    #[prost(string, repeated, tag="4")]
    pub target: ::std::vec::Vec<std::string::String>,
    /// Options for the run call.
    #[prost(message, optional, tag="5")]
    pub options: ::std::option::Option<RunOptions>,
    /// Partial run handle (optional). If specified, this will be a partial run
    /// execution, run up to the specified fetches.
    #[prost(string, tag="6")]
    pub partial_run_handle: std::string::String,
    /// If true then some errors, e.g., execution errors that have long
    /// error messages, may return an OK RunStepResponse with the actual
    /// error saved in the status_code/status_error_message fields of the
    /// response body. This is a workaround since the RPC subsystem may
    /// truncate long metadata messages.
    #[prost(bool, tag="7")]
    pub store_errors_in_response_body: bool,
    /// Unique identifier for this request. Every RunStepRequest must
    /// have a unique request_id, and retried RunStepRequest must have
    /// the same request_id. If request_id is zero, retry detection is disabled.
    #[prost(int64, tag="8")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunStepResponse {
    /// NOTE: The order of the returned tensors may or may not match
    /// the fetch order specified in RunStepRequest.
    #[prost(message, repeated, tag="1")]
    pub tensor: ::std::vec::Vec<NamedTensorProto>,
    /// Returned metadata if requested in the options.
    #[prost(message, optional, tag="2")]
    pub metadata: ::std::option::Option<RunMetadata>,
    /// If store_errors_in_response_body is true in the request, then
    /// optionally the server may return an OK status for the RPC and
    /// fill the true status into the fields below, to allow for messages
    /// that are too long to fit in metadata.
    #[prost(enumeration="error::Code", tag="3")]
    pub status_code: i32,
    #[prost(string, tag="4")]
    pub status_error_message: std::string::String,
}
////////////////////////////////////////////////////////////////////////////////
//
// PartialRunSetup method request/response protos.
//
// The caller should provide the future partial run feeds, fetches, and targets.
// Then the caller can use RunStepRequest with is_partial set to make partial
// run calls.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PartialRunSetupRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// Tensors to be fed in future steps.
    #[prost(string, repeated, tag="2")]
    pub feed: ::std::vec::Vec<std::string::String>,
    /// Fetches. A list of tensor names. The caller expects a tensor to be returned
    /// for each fetch[i] (see RunStepResponse.tensor), for corresponding partial
    /// RunStepRequests. The order of specified fetches does not change the
    /// execution order.
    #[prost(string, repeated, tag="3")]
    pub fetch: ::std::vec::Vec<std::string::String>,
    /// Target Nodes. A list of node names. The named nodes will be run in future
    /// steps, but their outputs will not be fetched.
    #[prost(string, repeated, tag="4")]
    pub target: ::std::vec::Vec<std::string::String>,
    /// Unique identifier for this request. Every PartialRunSetupRequest must
    /// have a unique request_id, and retried PartialRunSetupRequest must have
    /// the same request_id. If request_id is zero, retry detection is disabled.
    #[prost(int64, tag="5")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PartialRunSetupResponse {
    /// The unique handle corresponding to the ongoing partial run call setup by
    /// the invocation to PartialRunSetup. This handle may be passed to
    /// RunStepRequest to send and receive tensors for this partial run.
    #[prost(string, tag="1")]
    pub partial_run_handle: std::string::String,
}
////////////////////////////////////////////////////////////////////////////////
//
// CloseSession method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CloseSessionRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CloseSessionResponse {
}
/// Reset() allows misbehaving or slow sessions to be aborted and closed, and
/// causes their resources eventually to be released.  Reset() does not wait
/// for the computations in old sessions to cease; it merely starts the
/// process of tearing them down.  However, if a new session is started after
/// a Reset(), the new session is isolated from changes that old sessions
/// (started prior to the Reset()) may continue to make to resources, provided
/// all those resources are in containers listed in "containers".
///
/// Old sessions may continue to have side-effects on resources not in
/// containers listed in "containers", and thus may affect future
/// sessions' results in ways that are hard to predict.  Thus, if well-defined
/// behavior is desired, is it recommended that all containers be listed in
/// "containers".  Similarly, if a device_filter is specified, results may be
/// hard to predict.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResetRequest {
    /// A list of container names, which may be empty.
    ///
    /// If 'container' is not empty, releases resources in the given
    /// containers in all devices.
    ///
    /// If 'container' is empty, releases resources in the default
    /// container in all devices.
    #[prost(string, repeated, tag="1")]
    pub container: ::std::vec::Vec<std::string::String>,
    /// When any filters are present, only devices that match the filters
    /// will be reset. Each filter can be partially specified,
    /// e.g. "/job:ps" "/job:worker/replica:3", etc.
    #[prost(string, repeated, tag="2")]
    pub device_filters: ::std::vec::Vec<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResetResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// ListDevices method request/response protos.
//
// Returns information about the TensorFlow devices that are available
// to this master.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListDevicesRequest {
    /// Optional: session_handle must be returned by a CreateSession call to the
    /// same master service.
    ///
    /// When session_handle is empty, the ClusterSpec provided when the master was
    /// started is used to compute the available devices. If the session_handle is
    /// provided but not recognized, an error is returned. Finally, if a valid
    /// session_handle is provided, the cluster configuration for that session is
    /// used when computing the response.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListDevicesResponse {
    #[prost(message, repeated, tag="1")]
    pub local_device: ::std::vec::Vec<DeviceAttributes>,
    #[prost(message, repeated, tag="2")]
    pub remote_device: ::std::vec::Vec<DeviceAttributes>,
}
////////////////////////////////////////////////////////////////////////////////
//
// MakeCallable method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MakeCallableRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// Options that define the behavior of the created callable.
    #[prost(message, optional, tag="2")]
    pub options: ::std::option::Option<CallableOptions>,
    /// Unique identifier for this request. Every MakeCallableRequest must
    /// have a unique request_id, and retried MakeCallableRequest must have
    /// the same request_id. If request_id is zero, retry detection is disabled.
    #[prost(int64, tag="3")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MakeCallableResponse {
    /// A handle to the created callable.
    #[prost(int64, tag="1")]
    pub handle: i64,
}
////////////////////////////////////////////////////////////////////////////////
//
// RunCallable method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunCallableRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// REQUIRED: handle must be returned by a MakeCallable call to the same
    /// master service.
    #[prost(int64, tag="2")]
    pub handle: i64,
    /// Values of the tensors passed as arguments to the callable, in the order
    /// defined in the CallableOptions.feed field passed to MakeCallable.
    #[prost(message, repeated, tag="3")]
    pub feed: ::std::vec::Vec<TensorProto>,
    /// Unique identifier for this request. Every RunCallableRequest must
    /// have a unique request_id, and retried RunCallableRequest must have
    /// the same request_id. If request_id is zero, retry detection is disabled.
    #[prost(int64, tag="4")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunCallableResponse {
    /// Values of the tensors returned by the callable, in the order defined in the
    /// CallableOptions.fetch field passed to MakeCallable.
    #[prost(message, repeated, tag="1")]
    pub fetch: ::std::vec::Vec<TensorProto>,
    /// Returned metadata if requested in the options.
    #[prost(message, optional, tag="2")]
    pub metadata: ::std::option::Option<RunMetadata>,
}
////////////////////////////////////////////////////////////////////////////////
//
// ReleaseCallable method request/response protos.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReleaseCallableRequest {
    /// REQUIRED: session_handle must be returned by a CreateSession call
    /// to the same master service.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// REQUIRED: handle must be returned by a MakeCallable call to the same
    /// master service.
    #[prost(int64, tag="2")]
    pub handle: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReleaseCallableResponse {
}
// Control flow context related protocol buffers.

/// Protocol buffer representing the values in ControlFlowContext.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValuesDef {
    /// Value names that have been seen in this context.
    #[prost(string, repeated, tag="1")]
    pub values: ::std::vec::Vec<std::string::String>,
    /// Value names referenced by but external to this context.
    #[prost(map="string, string", tag="2")]
    pub external_values: ::std::collections::HashMap<std::string::String, std::string::String>,
}
/// Container for any kind of control flow context. Any other control flow
/// contexts that are added below should also be added here.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ControlFlowContextDef {
    #[prost(oneof="control_flow_context_def::Ctxt", tags="1, 2")]
    pub ctxt: ::std::option::Option<control_flow_context_def::Ctxt>,
}
pub mod control_flow_context_def {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Ctxt {
        #[prost(message, tag="1")]
        CondCtxt(super::CondContextDef),
        #[prost(message, tag="2")]
        WhileCtxt(super::WhileContextDef),
    }
}
/// Protocol buffer representing a CondContext object.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CondContextDef {
    /// Name of the context.
    #[prost(string, tag="1")]
    pub context_name: std::string::String,
    /// Name of the pred tensor.
    #[prost(string, tag="2")]
    pub pred_name: std::string::String,
    /// Name of the pivot tensor.
    #[prost(string, tag="3")]
    pub pivot_name: std::string::String,
    /// Branch prediction. 0 or 1.
    #[prost(int32, tag="4")]
    pub branch: i32,
    /// Values and external values in control flow context.
    #[prost(message, optional, tag="5")]
    pub values_def: ::std::option::Option<ValuesDef>,
    /// Contexts contained inside this context (e.g. nested conds).
    #[prost(message, repeated, tag="6")]
    pub nested_contexts: ::std::vec::Vec<ControlFlowContextDef>,
}
/// Protocol buffer representing a WhileContext object.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WhileContextDef {
    /// Name of the context.
    #[prost(string, tag="1")]
    pub context_name: std::string::String,
    /// The number of iterations allowed to run in parallel.
    #[prost(int32, tag="2")]
    pub parallel_iterations: i32,
    /// Whether backprop is enabled for this while loop.
    #[prost(bool, tag="3")]
    pub back_prop: bool,
    /// Whether GPU-CPU memory swap is enabled for this loop.
    #[prost(bool, tag="4")]
    pub swap_memory: bool,
    /// Name of the pivot tensor.
    #[prost(string, tag="5")]
    pub pivot_name: std::string::String,
    /// Name of the pivot_for_pred tensor.
    #[prost(string, tag="6")]
    pub pivot_for_pred_name: std::string::String,
    /// Name of the pivot_for_body tensor.
    #[prost(string, tag="7")]
    pub pivot_for_body_name: std::string::String,
    /// List of names for exit tensors.
    #[prost(string, repeated, tag="8")]
    pub loop_exit_names: ::std::vec::Vec<std::string::String>,
    /// List of names for enter tensors.
    #[prost(string, repeated, tag="10")]
    pub loop_enter_names: ::std::vec::Vec<std::string::String>,
    /// Values and external values in control flow context.
    #[prost(message, optional, tag="9")]
    pub values_def: ::std::option::Option<ValuesDef>,
    /// Optional name of the maximum_iterations tensor.
    #[prost(string, tag="11")]
    pub maximum_iterations_name: std::string::String,
    /// Contexts contained inside this context (e.g. nested whiles).
    #[prost(message, repeated, tag="12")]
    pub nested_contexts: ::std::vec::Vec<ControlFlowContextDef>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphDebugInfo {
    /// This stores all the source code file names and can be indexed by the
    /// `file_index`.
    #[prost(string, repeated, tag="1")]
    pub files: ::std::vec::Vec<std::string::String>,
    /// This maps a node name to a stack trace in the source code.
    #[prost(map="string, message", tag="2")]
    pub traces: ::std::collections::HashMap<std::string::String, graph_debug_info::StackTrace>,
}
pub mod graph_debug_info {
    /// This represents a file/line location in the source code.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FileLineCol {
        /// File name index, which can be used to retrieve the file name string from
        /// `files`. The value should be between 0 and (len(files)-1)
        #[prost(int32, tag="1")]
        pub file_index: i32,
        /// Line number in the file.
        #[prost(int32, tag="2")]
        pub line: i32,
        /// Col number in the file line.
        #[prost(int32, tag="3")]
        pub col: i32,
        /// Name of function contains the file line.
        #[prost(string, tag="4")]
        pub func: std::string::String,
        /// Source code contained in this file line.
        #[prost(string, tag="5")]
        pub code: std::string::String,
    }
    /// This represents a stack trace which is a ordered list of `FileLineCol`.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct StackTrace {
        /// Each line in the stack trace.
        #[prost(message, repeated, tag="1")]
        pub file_line_cols: ::std::vec::Vec<FileLineCol>,
    }
}
// A TensorBundle addition which saves extra information about the objects which
// own variables, allowing for more robust checkpoint loading into modified
// programs.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrackableObjectGraph {
    #[prost(message, repeated, tag="1")]
    pub nodes: ::std::vec::Vec<trackable_object_graph::TrackableObject>,
}
pub mod trackable_object_graph {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TrackableObject {
        /// Objects which this object depends on.
        #[prost(message, repeated, tag="1")]
        pub children: ::std::vec::Vec<trackable_object::ObjectReference>,
        /// Serialized data specific to this object.
        #[prost(message, repeated, tag="2")]
        pub attributes: ::std::vec::Vec<trackable_object::SerializedTensor>,
        /// Slot variables owned by this object.
        #[prost(message, repeated, tag="3")]
        pub slot_variables: ::std::vec::Vec<trackable_object::SlotVariableReference>,
    }
    pub mod trackable_object {
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct ObjectReference {
            /// An index into `TrackableObjectGraph.nodes`, indicating the object
            /// being referenced.
            #[prost(int32, tag="1")]
            pub node_id: i32,
            /// A user-provided name for the edge.
            #[prost(string, tag="2")]
            pub local_name: std::string::String,
        }
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct SerializedTensor {
            /// A name for the Tensor. Simple variables have only one
            /// `SerializedTensor` named "VARIABLE_VALUE" by convention. This value may
            /// be restored on object creation as an optimization.
            #[prost(string, tag="1")]
            pub name: std::string::String,
            /// The full name of the variable/tensor, if applicable. Used to allow
            /// name-based loading of checkpoints which were saved using an
            /// object-based API. Should match the checkpoint key which would have been
            /// assigned by tf.train.Saver.
            #[prost(string, tag="2")]
            pub full_name: std::string::String,
            /// The generated name of the Tensor in the checkpoint.
            #[prost(string, tag="3")]
            pub checkpoint_key: std::string::String,
            /// Whether checkpoints should be considered as matching even without this
            /// value restored. Used for non-critical values which don't affect the
            /// TensorFlow graph, such as layer configurations.
            #[prost(bool, tag="4")]
            pub optional_restore: bool,
        }
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct SlotVariableReference {
            /// An index into `TrackableObjectGraph.nodes`, indicating the
            /// variable object this slot was created for.
            #[prost(int32, tag="1")]
            pub original_variable_node_id: i32,
            /// The name of the slot (e.g. "m"/"v").
            #[prost(string, tag="2")]
            pub slot_name: std::string::String,
            /// An index into `TrackableObjectGraph.nodes`, indicating the
            /// `Object` with the value of the slot variable.
            #[prost(int32, tag="3")]
            pub slot_variable_node_id: i32,
        }
    }
}
/// `StructuredValue` represents a dynamically typed value representing various
/// data structures that are inspired by Python data structures typically used in
/// TensorFlow functions as inputs and outputs.
///
/// For example when saving a Layer there may be a `training` argument. If the
/// user passes a boolean True/False, that switches between two concrete
/// TensorFlow functions. In order to switch between them in the same way after
/// loading the SavedModel, we need to represent "True" and "False".
///
/// A more advanced example might be a function which takes a list of
/// dictionaries mapping from strings to Tensors. In order to map from
/// user-specified arguments `[{"a": tf.constant(1.)}, {"q": tf.constant(3.)}]`
/// after load to the right saved TensorFlow function, we need to represent the
/// nested structure and the strings, recording that we have a trace for anything
/// matching `[{"a": tf.TensorSpec(None, tf.float32)}, {"q": tf.TensorSpec([],
/// tf.float64)}]` as an example.
///
/// Likewise functions may return nested structures of Tensors, for example
/// returning a dictionary mapping from strings to Tensors. In order for the
/// loaded function to return the same structure we need to serialize it.
///
/// This is an ergonomic aid for working with loaded SavedModels, not a promise
/// to serialize all possible function signatures. For example we do not expect
/// to pickle generic Python objects, and ideally we'd stay language-agnostic.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StructuredValue {
    /// The kind of value.
    #[prost(oneof="structured_value::Kind", tags="1, 11, 12, 13, 14, 31, 32, 33, 34, 51, 52, 53, 54")]
    pub kind: ::std::option::Option<structured_value::Kind>,
}
pub mod structured_value {
    /// The kind of value.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Kind {
        /// Represents None.
        #[prost(message, tag="1")]
        NoneValue(super::NoneValue),
        /// Represents a double-precision floating-point value (a Python `float`).
        #[prost(double, tag="11")]
        Float64Value(f64),
        /// Represents a signed integer value, limited to 64 bits.
        /// Larger values from Python's arbitrary-precision integers are unsupported.
        #[prost(sint64, tag="12")]
        Int64Value(i64),
        /// Represents a string of Unicode characters stored in a Python `str`.
        /// In Python 3, this is exactly what type `str` is.
        /// In Python 2, this is the UTF-8 encoding of the characters.
        /// For strings with ASCII characters only (as often used in TensorFlow code)
        /// there is effectively no difference between the language versions.
        /// The obsolescent `unicode` type of Python 2 is not supported here.
        #[prost(string, tag="13")]
        StringValue(std::string::String),
        /// Represents a boolean value.
        #[prost(bool, tag="14")]
        BoolValue(bool),
        /// Represents a TensorShape.
        #[prost(message, tag="31")]
        TensorShapeValue(super::TensorShapeProto),
        /// Represents an enum value for dtype.
        #[prost(enumeration="super::DataType", tag="32")]
        TensorDtypeValue(i32),
        /// Represents a value for tf.TensorSpec.
        #[prost(message, tag="33")]
        TensorSpecValue(super::TensorSpecProto),
        /// Represents a value for tf.TypeSpec.
        #[prost(message, tag="34")]
        TypeSpecValue(Box<super::TypeSpecProto>),
        /// Represents a list of `Value`.
        #[prost(message, tag="51")]
        ListValue(super::ListValue),
        /// Represents a tuple of `Value`.
        #[prost(message, tag="52")]
        TupleValue(super::TupleValue),
        /// Represents a dict `Value`.
        #[prost(message, tag="53")]
        DictValue(super::DictValue),
        /// Represents Python's namedtuple.
        #[prost(message, tag="54")]
        NamedTupleValue(super::NamedTupleValue),
    }
}
/// Represents None.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NoneValue {
}
/// Represents a Python list.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListValue {
    #[prost(message, repeated, tag="1")]
    pub values: ::std::vec::Vec<StructuredValue>,
}
/// Represents a Python tuple.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TupleValue {
    #[prost(message, repeated, tag="1")]
    pub values: ::std::vec::Vec<StructuredValue>,
}
/// Represents a Python dict keyed by `str`.
/// The comment on Unicode from Value.string_value applies analogously.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DictValue {
    #[prost(map="string, message", tag="1")]
    pub fields: ::std::collections::HashMap<std::string::String, StructuredValue>,
}
/// Represents a (key, value) pair.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PairValue {
    #[prost(string, tag="1")]
    pub key: std::string::String,
    #[prost(message, optional, tag="2")]
    pub value: ::std::option::Option<StructuredValue>,
}
/// Represents Python's namedtuple.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NamedTupleValue {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(message, repeated, tag="2")]
    pub values: ::std::vec::Vec<PairValue>,
}
/// A protobuf to tf.TensorSpec.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorSpecProto {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    #[prost(enumeration="DataType", tag="3")]
    pub dtype: i32,
}
/// Represents a tf.TypeSpec
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeSpecProto {
    #[prost(enumeration="type_spec_proto::TypeSpecClass", tag="1")]
    pub type_spec_class: i32,
    /// The value returned by TypeSpec._serialize().
    #[prost(message, optional, boxed, tag="2")]
    pub type_state: ::std::option::Option<::std::boxed::Box<StructuredValue>>,
    /// This is currently redundant with the type_spec_class enum, and is only
    /// used for error reporting.  In particular, if you use an older binary to
    /// load a newer model, and the model uses a TypeSpecClass that the older
    /// binary doesn't support, then this lets us display a useful error message.
    #[prost(string, tag="3")]
    pub type_spec_class_name: std::string::String,
}
pub mod type_spec_proto {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum TypeSpecClass {
        Unknown = 0,
        /// tf.SparseTensorSpec
        SparseTensorSpec = 1,
        /// tf.IndexedSlicesSpec
        IndexedSlicesSpec = 2,
        /// tf.RaggedTensorSpec
        RaggedTensorSpec = 3,
        /// tf.TensorArraySpec
        TensorArraySpec = 4,
        /// tf.data.DatasetSpec
        DataDatasetSpec = 5,
        /// IteratorSpec from data/ops/iterator_ops.py
        DataIteratorSpec = 6,
        /// tf.OptionalSpec
        OptionalSpec = 7,
        /// PerReplicaSpec from distribute/values.py
        PerReplicaSpec = 8,
    }
}
/// Protocol buffer representing a Variable.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VariableDef {
    /// Name of the variable tensor.
    #[prost(string, tag="1")]
    pub variable_name: std::string::String,
    /// Name of the tensor holding the variable's initial value.
    #[prost(string, tag="6")]
    pub initial_value_name: std::string::String,
    /// Name of the initializer op.
    #[prost(string, tag="2")]
    pub initializer_name: std::string::String,
    /// Name of the snapshot tensor.
    #[prost(string, tag="3")]
    pub snapshot_name: std::string::String,
    /// Support for saving variables as slices of a larger variable.
    #[prost(message, optional, tag="4")]
    pub save_slice_info_def: ::std::option::Option<SaveSliceInfoDef>,
    /// Whether to represent this as a ResourceVariable.
    #[prost(bool, tag="5")]
    pub is_resource: bool,
    /// Whether this variable should be trained.
    #[prost(bool, tag="7")]
    pub trainable: bool,
    /// Indicates when a distributed variable will be synced.
    #[prost(enumeration="VariableSynchronization", tag="8")]
    pub synchronization: i32,
    /// Indicates how a distributed variable will be aggregated.
    #[prost(enumeration="VariableAggregation", tag="9")]
    pub aggregation: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SaveSliceInfoDef {
    /// Name of the full variable of which this is a slice.
    #[prost(string, tag="1")]
    pub full_name: std::string::String,
    /// Shape of the full variable.
    #[prost(int64, repeated, tag="2")]
    pub full_shape: ::std::vec::Vec<i64>,
    /// Offset of this variable into the full variable.
    #[prost(int64, repeated, tag="3")]
    pub var_offset: ::std::vec::Vec<i64>,
    /// Shape of this variable.
    #[prost(int64, repeated, tag="4")]
    pub var_shape: ::std::vec::Vec<i64>,
}
/// Indicates when a distributed variable will be synced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum VariableSynchronization {
    /// `AUTO`: Indicates that the synchronization will be determined by the
    /// current `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    /// `ON_WRITE`).
    Auto = 0,
    /// `NONE`: Indicates that there will only be one copy of the variable, so
    /// there is no need to sync.
    None = 1,
    /// `ON_WRITE`: Indicates that the variable will be updated across devices
    /// every time it is written.
    OnWrite = 2,
    /// `ON_READ`: Indicates that the variable will be aggregated across devices
    /// when it is read (eg. when checkpointing or when evaluating an op that uses
    /// the variable).
    OnRead = 3,
}
/// Indicates how a distributed variable will be aggregated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum VariableAggregation {
    /// `NONE`: This is the default, giving an error if you use a
    /// variable-update operation with multiple replicas.
    None = 0,
    /// `SUM`: Add the updates across replicas.
    Sum = 1,
    /// `MEAN`: Take the arithmetic mean ("average") of the updates across
    /// replicas.
    Mean = 2,
    /// `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
    /// update, but we only want to perform the update once. Used, e.g., for the
    /// global step counter.
    OnlyFirstReplica = 3,
}
// A SavedObjectGraph is part of object-based SavedModels in TF 2.0. It
// describes the directed graph of Python objects (or equivalent in other
// languages) that make up a model, with nodes[0] at the root.

// SavedObjectGraph shares some structure with TrackableObjectGraph, but
// SavedObjectGraph belongs to the MetaGraph and contains pointers to functions
// and type information, while TrackableObjectGraph lives in the checkpoint
// and contains pointers only to variable values.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedObjectGraph {
    /// Flattened list of objects in the object graph.
    ///
    /// The position of the object in this list indicates its id.
    /// Nodes[0] is considered the root node.
    #[prost(message, repeated, tag="1")]
    pub nodes: ::std::vec::Vec<SavedObject>,
    /// Information about captures and output structures in concrete functions.
    /// Referenced from SavedBareConcreteFunction and SavedFunction.
    #[prost(map="string, message", tag="2")]
    pub concrete_functions: ::std::collections::HashMap<std::string::String, SavedConcreteFunction>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedObject {
    /// Objects which this object depends on: named edges in the dependency
    /// graph.
    ///
    /// Note: currently only valid if kind == "user_object".
    #[prost(message, repeated, tag="1")]
    pub children: ::std::vec::Vec<trackable_object_graph::trackable_object::ObjectReference>,
    /// Slot variables owned by this object. This describes the three-way
    /// (optimizer, variable, slot variable) relationship; none of the three
    /// depend on the others directly.
    ///
    /// Note: currently only valid if kind == "user_object".
    #[prost(message, repeated, tag="3")]
    pub slot_variables: ::std::vec::Vec<trackable_object_graph::trackable_object::SlotVariableReference>,
    #[prost(oneof="saved_object::Kind", tags="4, 5, 6, 7, 8, 9, 10")]
    pub kind: ::std::option::Option<saved_object::Kind>,
}
pub mod saved_object {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Kind {
        #[prost(message, tag="4")]
        UserObject(super::SavedUserObject),
        #[prost(message, tag="5")]
        Asset(super::SavedAsset),
        #[prost(message, tag="6")]
        Function(super::SavedFunction),
        #[prost(message, tag="7")]
        Variable(super::SavedVariable),
        #[prost(message, tag="8")]
        BareConcreteFunction(super::SavedBareConcreteFunction),
        #[prost(message, tag="9")]
        Constant(super::SavedConstant),
        #[prost(message, tag="10")]
        Resource(super::SavedResource),
    }
}
/// A SavedUserObject is an object (in the object-oriented language of the
/// TensorFlow program) of some user- or framework-defined class other than
/// those handled specifically by the other kinds of SavedObjects.
///
/// This object cannot be evaluated as a tensor, and therefore cannot be bound
/// to an input of a function.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedUserObject {
    /// Corresponds to a registration of the type to use in the loading program.
    #[prost(string, tag="1")]
    pub identifier: std::string::String,
    /// Version information from the producer of this SavedUserObject.
    #[prost(message, optional, tag="2")]
    pub version: ::std::option::Option<VersionDef>,
    /// Initialization-related metadata.
    #[prost(string, tag="3")]
    pub metadata: std::string::String,
}
/// A SavedAsset points to an asset in the MetaGraph.
///
/// When bound to a function this object evaluates to a tensor with the absolute
/// filename. Users should not depend on a particular part of the filename to
/// remain stable (e.g. basename could be changed).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedAsset {
    /// Index into `MetaGraphDef.asset_file_def[]` that describes the Asset.
    ///
    /// Only the field `AssetFileDef.filename` is used. Other fields, such as
    /// `AssetFileDef.tensor_info`, MUST be ignored.
    #[prost(int32, tag="1")]
    pub asset_file_def_index: i32,
}
/// A function with multiple signatures, possibly with non-Tensor arguments.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedFunction {
    #[prost(string, repeated, tag="1")]
    pub concrete_functions: ::std::vec::Vec<std::string::String>,
    #[prost(message, optional, tag="2")]
    pub function_spec: ::std::option::Option<FunctionSpec>,
}
/// Stores low-level information about a concrete function. Referenced in either
/// a SavedFunction or a SavedBareConcreteFunction.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedConcreteFunction {
    /// Bound inputs to the function. The SavedObjects identified by the node ids
    /// given here are appended as extra inputs to the caller-supplied inputs.
    /// The only types of SavedObjects valid here are SavedVariable, SavedResource
    /// and SavedAsset.
    #[prost(int32, repeated, tag="2")]
    pub bound_inputs: ::std::vec::Vec<i32>,
    /// Input in canonicalized form that was received to create this concrete
    /// function.
    #[prost(message, optional, tag="3")]
    pub canonicalized_input_signature: ::std::option::Option<StructuredValue>,
    /// Output that was the return value of this function after replacing all
    /// Tensors with TensorSpecs. This can be an arbitrary nested function and will
    /// be used to reconstruct the full structure from pure tensors.
    #[prost(message, optional, tag="4")]
    pub output_signature: ::std::option::Option<StructuredValue>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedBareConcreteFunction {
    /// Identifies a SavedConcreteFunction.
    #[prost(string, tag="1")]
    pub concrete_function_name: std::string::String,
    /// A sequence of unique strings, one per Tensor argument.
    #[prost(string, repeated, tag="2")]
    pub argument_keywords: ::std::vec::Vec<std::string::String>,
    /// The prefix of `argument_keywords` which may be identified by position.
    #[prost(int64, tag="3")]
    pub allowed_positional_arguments: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedConstant {
    /// An Operation name for a ConstantOp in this SavedObjectGraph's MetaGraph.
    #[prost(string, tag="1")]
    pub operation: std::string::String,
}
/// Represents a Variable that is initialized by loading the contents from the
/// checkpoint.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedVariable {
    #[prost(enumeration="DataType", tag="1")]
    pub dtype: i32,
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    #[prost(bool, tag="3")]
    pub trainable: bool,
    #[prost(enumeration="VariableSynchronization", tag="4")]
    pub synchronization: i32,
    #[prost(enumeration="VariableAggregation", tag="5")]
    pub aggregation: i32,
    #[prost(string, tag="6")]
    pub name: std::string::String,
}
/// Represents `FunctionSpec` used in `Function`. This represents a
/// function that has been wrapped as a TensorFlow `Function`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FunctionSpec {
    /// Full arg spec from inspect.getfullargspec().
    #[prost(message, optional, tag="1")]
    pub fullargspec: ::std::option::Option<StructuredValue>,
    /// Whether this represents a class method.
    #[prost(bool, tag="2")]
    pub is_method: bool,
    /// The input signature, if specified.
    #[prost(message, optional, tag="5")]
    pub input_signature: ::std::option::Option<StructuredValue>,
}
/// A SavedResource represents a TF object that holds state during its lifetime.
/// An object of this type can have a reference to a:
/// create_resource() and an initialize() function.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedResource {
    /// A device specification indicating a required placement for the resource
    /// creation function, e.g. "CPU". An empty string allows the user to select a
    /// device.
    #[prost(string, tag="1")]
    pub device: std::string::String,
}
/// NOTE: This protocol buffer is evolving, and will go through revisions in the
/// coming months.
///
/// Protocol buffer containing the following which are necessary to restart
/// training, run inference. It can be used to serialize/de-serialize memory
/// objects necessary for running computation in a graph when crossing the
/// process boundary. It can be used for long term storage of graphs,
/// cross-language execution of graphs, etc.
///   MetaInfoDef
///   GraphDef
///   SaverDef
///   CollectionDef
///   TensorInfo
///   SignatureDef
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MetaGraphDef {
    #[prost(message, optional, tag="1")]
    pub meta_info_def: ::std::option::Option<meta_graph_def::MetaInfoDef>,
    /// GraphDef.
    #[prost(message, optional, tag="2")]
    pub graph_def: ::std::option::Option<GraphDef>,
    /// SaverDef.
    #[prost(message, optional, tag="3")]
    pub saver_def: ::std::option::Option<SaverDef>,
    /// collection_def: Map from collection name to collections.
    /// See CollectionDef section for details.
    #[prost(map="string, message", tag="4")]
    pub collection_def: ::std::collections::HashMap<std::string::String, CollectionDef>,
    /// signature_def: Map from user supplied key for a signature to a single
    /// SignatureDef.
    #[prost(map="string, message", tag="5")]
    pub signature_def: ::std::collections::HashMap<std::string::String, SignatureDef>,
    /// Asset file def to be used with the defined graph.
    #[prost(message, repeated, tag="6")]
    pub asset_file_def: ::std::vec::Vec<AssetFileDef>,
    /// Extra information about the structure of functions and stateful objects.
    #[prost(message, optional, tag="7")]
    pub object_graph_def: ::std::option::Option<SavedObjectGraph>,
}
pub mod meta_graph_def {
    /// Meta information regarding the graph to be exported.  To be used by users
    /// of this protocol buffer to encode information regarding their meta graph.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct MetaInfoDef {
        /// User specified Version string. Can be the name of the model and revision,
        /// steps this model has been trained to, etc.
        #[prost(string, tag="1")]
        pub meta_graph_version: std::string::String,
        /// A copy of the OpDefs used by the producer of this graph_def.
        /// Descriptions and Ops not used in graph_def are stripped out.
        #[prost(message, optional, tag="2")]
        pub stripped_op_list: ::std::option::Option<super::OpList>,
        /// A serialized protobuf. Can be the time this meta graph is created, or
        /// modified, or name of the model.
        #[prost(message, optional, tag="3")]
        pub any_info: ::std::option::Option<::prost_types::Any>,
        /// User supplied tag(s) on the meta_graph and included graph_def.
        ///
        /// MetaGraphDefs should be tagged with their capabilities or use-cases.
        /// Examples: "train", "serve", "gpu", "tpu", etc.
        /// These tags enable loaders to access the MetaGraph(s) appropriate for a
        /// specific use-case or runtime environment.
        #[prost(string, repeated, tag="4")]
        pub tags: ::std::vec::Vec<std::string::String>,
        /// The __version__ string of the tensorflow build used to write this graph.
        /// This will be populated by the framework, which will overwrite any user
        /// supplied value.
        #[prost(string, tag="5")]
        pub tensorflow_version: std::string::String,
        /// The __git_version__ string of the tensorflow build used to write this
        /// graph. This will be populated by the framework, which will overwrite any
        /// user supplied value.
        #[prost(string, tag="6")]
        pub tensorflow_git_version: std::string::String,
        /// A flag to denote whether default-valued attrs have been stripped from
        /// the nodes in this graph_def.
        #[prost(bool, tag="7")]
        pub stripped_default_attrs: bool,
    }
}
/// CollectionDef should cover most collections.
/// To add a user-defined collection, do one of the following:
/// 1. For simple data types, such as string, int, float:
///      tf.add_to_collection("your_collection_name", your_simple_value)
///    strings will be stored as bytes_list.
///
/// 2. For Protobuf types, there are three ways to add them:
///    1) tf.add_to_collection("your_collection_name",
///         your_proto.SerializeToString())
///
///       collection_def {
///         key: "user_defined_bytes_collection"
///         value {
///           bytes_list {
///             value: "queue_name: \"test_queue\"\n"
///           }
///         }
///       }
///
///  or
///
///    2) tf.add_to_collection("your_collection_name", str(your_proto))
///
///       collection_def {
///         key: "user_defined_string_collection"
///         value {
///          bytes_list {
///             value: "\n\ntest_queue"
///           }
///         }
///       }
///
///  or
///
///    3) any_buf = any_pb2.Any()
///       tf.add_to_collection("your_collection_name",
///         any_buf.Pack(your_proto))
///
///       collection_def {
///         key: "user_defined_any_collection"
///         value {
///           any_list {
///             value {
///               type_url: "type.googleapis.com/tensorflow.QueueRunnerDef"
///               value: "\n\ntest_queue"
///             }
///           }
///         }
///       }
///
/// 3. For Python objects, implement to_proto() and from_proto(), and register
///    them in the following manner:
///    ops.register_proto_function("your_collection_name",
///                                proto_type,
///                                to_proto=YourPythonObject.to_proto,
///                                from_proto=YourPythonObject.from_proto)
///    These functions will be invoked to serialize and de-serialize the
///    collection. For example,
///    ops.register_proto_function(ops.GraphKeys.GLOBAL_VARIABLES,
///                                proto_type=variable_pb2.VariableDef,
///                                to_proto=Variable.to_proto,
///                                from_proto=Variable.from_proto)
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CollectionDef {
    #[prost(oneof="collection_def::Kind", tags="1, 2, 3, 4, 5")]
    pub kind: ::std::option::Option<collection_def::Kind>,
}
pub mod collection_def {
    /// NodeList is used for collecting nodes in graph. For example
    /// collection_def {
    ///   key: "summaries"
    ///   value {
    ///     node_list {
    ///       value: "input_producer/ScalarSummary:0"
    ///       value: "shuffle_batch/ScalarSummary:0"
    ///       value: "ImageSummary:0"
    ///     }
    ///   }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct NodeList {
        #[prost(string, repeated, tag="1")]
        pub value: ::std::vec::Vec<std::string::String>,
    }
    /// BytesList is used for collecting strings and serialized protobufs. For
    /// example:
    /// collection_def {
    ///   key: "trainable_variables"
    ///   value {
    ///     bytes_list {
    ///       value: "\n\017conv1/weights:0\022\024conv1/weights/Assign
    ///              \032\024conv1/weights/read:0"
    ///       value: "\n\016conv1/biases:0\022\023conv1/biases/Assign\032
    ///              \023conv1/biases/read:0"
    ///     }
    ///   }
    /// }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct BytesList {
        #[prost(bytes, repeated, tag="1")]
        pub value: ::std::vec::Vec<std::vec::Vec<u8>>,
    }
    /// Int64List is used for collecting int, int64 and long values.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Int64List {
        #[prost(int64, repeated, tag="1")]
        pub value: ::std::vec::Vec<i64>,
    }
    /// FloatList is used for collecting float values.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FloatList {
        #[prost(float, repeated, tag="1")]
        pub value: ::std::vec::Vec<f32>,
    }
    /// AnyList is used for collecting Any protos.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AnyList {
        #[prost(message, repeated, tag="1")]
        pub value: ::std::vec::Vec<::prost_types::Any>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Kind {
        #[prost(message, tag="1")]
        NodeList(NodeList),
        #[prost(message, tag="2")]
        BytesList(BytesList),
        #[prost(message, tag="3")]
        Int64List(Int64List),
        #[prost(message, tag="4")]
        FloatList(FloatList),
        #[prost(message, tag="5")]
        AnyList(AnyList),
    }
}
/// Information about a Tensor necessary for feeding or retrieval.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorInfo {
    #[prost(enumeration="DataType", tag="2")]
    pub dtype: i32,
    /// The static shape should be recorded here, to the extent that it can
    /// be known in advance.  In the case of a SparseTensor, this field describes
    /// the logical shape of the represented tensor (aka dense_shape).
    #[prost(message, optional, tag="3")]
    pub tensor_shape: ::std::option::Option<TensorShapeProto>,
    #[prost(oneof="tensor_info::Encoding", tags="1, 4, 5")]
    pub encoding: ::std::option::Option<tensor_info::Encoding>,
}
pub mod tensor_info {
    /// For sparse tensors, The COO encoding stores a triple of values, indices,
    /// and shape.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct CooSparse {
        /// The shape of the values Tensor is [?].  Its dtype must be the dtype of
        /// the SparseTensor as a whole, given in the enclosing TensorInfo.
        #[prost(string, tag="1")]
        pub values_tensor_name: std::string::String,
        /// The indices Tensor must have dtype int64 and shape [?, ?].
        #[prost(string, tag="2")]
        pub indices_tensor_name: std::string::String,
        /// The dynamic logical shape represented by the SparseTensor is recorded in
        /// the Tensor referenced here.  It must have dtype int64 and shape [?].
        #[prost(string, tag="3")]
        pub dense_shape_tensor_name: std::string::String,
    }
    /// Generic encoding for composite tensors.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct CompositeTensor {
        /// The serialized TypeSpec for the composite tensor.
        #[prost(message, optional, tag="1")]
        pub type_spec: ::std::option::Option<super::TypeSpecProto>,
        /// A TensorInfo for each flattened component tensor.
        #[prost(message, repeated, tag="2")]
        pub components: ::std::vec::Vec<super::TensorInfo>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Encoding {
        /// For dense `Tensor`s, the name of the tensor in the graph.
        #[prost(string, tag="1")]
        Name(std::string::String),
        /// There are many possible encodings of sparse matrices
        /// (https://en.wikipedia.org/wiki/Sparse_matrix).  Currently, TensorFlow
        /// uses only the COO encoding.  This is supported and documented in the
        /// SparseTensor Python class.
        #[prost(message, tag="4")]
        CooSparse(CooSparse),
        /// Generic encoding for CompositeTensors.
        #[prost(message, tag="5")]
        CompositeTensor(CompositeTensor),
    }
}
/// SignatureDef defines the signature of a computation supported by a TensorFlow
/// graph.
///
/// For example, a model with two loss computations, sharing a single input,
/// might have the following signature_def map.
///
/// Note that across the two SignatureDefs "loss_A" and "loss_B", the input key,
/// output key, and method_name are identical, and will be used by system(s) that
/// implement or rely upon this particular loss method. The output tensor names
/// differ, demonstrating how different outputs can exist for the same method.
///
/// signature_def {
///   key: "loss_A"
///   value {
///     inputs {
///       key: "input"
///       value {
///         name: "input:0"
///         dtype: DT_STRING
///         tensor_shape: ...
///       }
///     }
///     outputs {
///       key: "loss_output"
///       value {
///         name: "loss_output_A:0"
///         dtype: DT_FLOAT
///         tensor_shape: ...
///       }
///     }
///   }
///   ...
///   method_name: "some/package/compute_loss"
/// }
/// signature_def {
///   key: "loss_B"
///   value {
///     inputs {
///       key: "input"
///       value {
///         name: "input:0"
///         dtype: DT_STRING
///         tensor_shape: ...
///       }
///     }
///     outputs {
///       key: "loss_output"
///       value {
///         name: "loss_output_B:0"
///         dtype: DT_FLOAT
///         tensor_shape: ...
///       }
///     }
///   }
///   ...
///   method_name: "some/package/compute_loss"
/// }
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SignatureDef {
    /// Named input parameters.
    #[prost(map="string, message", tag="1")]
    pub inputs: ::std::collections::HashMap<std::string::String, TensorInfo>,
    /// Named output parameters.
    #[prost(map="string, message", tag="2")]
    pub outputs: ::std::collections::HashMap<std::string::String, TensorInfo>,
    /// Extensible method_name information enabling third-party users to mark a
    /// SignatureDef as supporting a particular method. This enables producers and
    /// consumers of SignatureDefs, e.g. a model definition library and a serving
    /// library to have a clear hand-off regarding the semantics of a computation.
    ///
    /// Note that multiple SignatureDefs in a single MetaGraphDef may have the same
    /// method_name. This is commonly used to support multi-headed computation,
    /// where a single graph computation may return multiple results.
    #[prost(string, tag="3")]
    pub method_name: std::string::String,
}
/// An asset file def for a single file or a set of sharded files with the same
/// name.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AssetFileDef {
    /// The tensor to bind the asset filename to.
    #[prost(message, optional, tag="1")]
    pub tensor_info: ::std::option::Option<TensorInfo>,
    /// The filename within an assets directory. Note: does not include the path
    /// prefix, i.e. directories. For an asset at /tmp/path/vocab.txt, the filename
    /// would be "vocab.txt".
    #[prost(string, tag="2")]
    pub filename: std::string::String,
}
/// SavedModel is the high level serialization format for TensorFlow Models.
/// See [todo: doc links, similar to session_bundle] for more information.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SavedModel {
    /// The schema version of the SavedModel instance. Used for versioning when
    /// making future changes to the specification/implementation. Initial value
    /// at release will be 1.
    #[prost(int64, tag="1")]
    pub saved_model_schema_version: i64,
    /// One or more MetaGraphs.
    #[prost(message, repeated, tag="2")]
    pub meta_graphs: ::std::vec::Vec<MetaGraphDef>,
}
/// Defines the configuration of a single TensorFlow server.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServerDef {
    /// The cluster of which this server is a member.
    #[prost(message, optional, tag="1")]
    pub cluster: ::std::option::Option<ClusterDef>,
    /// The name of the job of which this server is a member.
    ///
    /// NOTE(mrry): The `cluster` field must contain a `JobDef` with a `name` field
    /// that matches this name.
    #[prost(string, tag="2")]
    pub job_name: std::string::String,
    /// The task index of this server in its job.
    ///
    /// NOTE: The `cluster` field must contain a `JobDef` with a matching `name`
    /// and a mapping in its `tasks` field for this index.
    #[prost(int32, tag="3")]
    pub task_index: i32,
    /// The default configuration for sessions that run on this server.
    #[prost(message, optional, tag="4")]
    pub default_session_config: ::std::option::Option<ConfigProto>,
    /// The protocol to be used by this server.
    ///
    /// Acceptable values include: "grpc", "grpc+verbs".
    #[prost(string, tag="5")]
    pub protocol: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CudnnVersion {
    #[prost(int32, tag="1")]
    pub major: i32,
    #[prost(int32, tag="2")]
    pub minor: i32,
    #[prost(int32, tag="3")]
    pub patch: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputeCapability {
    #[prost(int32, tag="1")]
    pub major: i32,
    #[prost(int32, tag="2")]
    pub minor: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AutotuneResult {
    #[prost(int64, tag="8")]
    pub scratch_bytes: i64,
    #[prost(message, optional, tag="9")]
    pub run_time: ::std::option::Option<::prost_types::Duration>,
    #[prost(message, optional, tag="7")]
    pub failure: ::std::option::Option<autotune_result::FailureResult>,
    #[prost(oneof="autotune_result::Key", tags="5, 6")]
    pub key: ::std::option::Option<autotune_result::Key>,
}
pub mod autotune_result {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FailureResult {
        #[prost(enumeration="FailureKind", tag="1")]
        pub kind: i32,
        #[prost(string, tag="2")]
        pub msg: std::string::String,
        #[prost(int64, tag="13")]
        pub buffer_address: i64,
        /// For failure_kind == WRONG_RESULT, this field indicates the reference
        /// configuration that we compared against.
        ///
        /// Note that the reference algorithm isn't always correct.  However,
        /// empirically it's more correct, as it's "algo 0", less fancy than the
        /// compared one.
        #[prost(oneof="failure_result::Key", tags="11, 12")]
        pub key: ::std::option::Option<failure_result::Key>,
    }
    pub mod failure_result {
        /// For failure_kind == WRONG_RESULT, this field indicates the reference
        /// configuration that we compared against.
        ///
        /// Note that the reference algorithm isn't always correct.  However,
        /// empirically it's more correct, as it's "algo 0", less fancy than the
        /// compared one.
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Key {
            #[prost(message, tag="11")]
            ReferenceConv(super::ConvKey),
            #[prost(message, tag="12")]
            ReferenceGemm(super::GemmKey),
        }
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ConvKey {
        #[prost(int64, tag="1")]
        pub algorithm: i64,
        #[prost(bool, tag="2")]
        pub tensor_ops_enabled: bool,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GemmKey {
        #[prost(int64, tag="1")]
        pub algorithm: i64,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum FailureKind {
        Unknown = 0,
        RedzoneModified = 1,
        WrongResult = 2,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Key {
        #[prost(message, tag="5")]
        Conv(ConvKey),
        #[prost(message, tag="6")]
        Gemm(GemmKey),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AutotuningLog {
    #[prost(message, optional, tag="1")]
    pub instr: ::std::option::Option<::prost_types::Any>,
    /// Records all auto-tuning results per algorithm.
    #[prost(message, repeated, tag="2")]
    pub results: ::std::vec::Vec<AutotuneResult>,
    #[prost(message, optional, tag="3")]
    pub cudnn_version: ::std::option::Option<CudnnVersion>,
    #[prost(message, optional, tag="4")]
    pub compute_capability: ::std::option::Option<ComputeCapability>,
    /// stream_executor::DeviceDescription::pci_bus_id.
    #[prost(string, tag="5")]
    pub device_pci_bus_id: std::string::String,
    #[prost(string, tag="6")]
    pub blas_version: std::string::String,
}
/// Records the creation of a new replay session.  We record the device listing
/// here to capture the state of the cluster.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NewReplaySession {
    #[prost(message, optional, tag="1")]
    pub devices: ::std::option::Option<ListDevicesResponse>,
    #[prost(string, tag="2")]
    pub session_handle: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReplayOp {
    #[prost(double, tag="31")]
    pub start_time_us: f64,
    #[prost(double, tag="32")]
    pub end_time_us: f64,
    #[prost(oneof="replay_op::Op", tags="1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")]
    pub op: ::std::option::Option<replay_op::Op>,
    #[prost(oneof="replay_op::Response", tags="21, 22, 23, 24, 25, 26, 27, 28, 29, 30")]
    pub response: ::std::option::Option<replay_op::Response>,
}
pub mod replay_op {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Op {
        #[prost(message, tag="1")]
        CreateSession(super::CreateSessionRequest),
        #[prost(message, tag="2")]
        ExtendSession(super::ExtendSessionRequest),
        #[prost(message, tag="3")]
        PartialRunSetup(super::PartialRunSetupRequest),
        #[prost(message, tag="4")]
        RunStep(super::RunStepRequest),
        #[prost(message, tag="5")]
        CloseSession(super::CloseSessionRequest),
        #[prost(message, tag="6")]
        ListDevices(super::ListDevicesRequest),
        #[prost(message, tag="7")]
        ResetRequest(super::ResetRequest),
        #[prost(message, tag="8")]
        MakeCallable(super::MakeCallableRequest),
        #[prost(message, tag="9")]
        RunCallable(super::RunCallableRequest),
        #[prost(message, tag="10")]
        ReleaseCallable(super::ReleaseCallableRequest),
        #[prost(message, tag="11")]
        NewReplaySession(super::NewReplaySession),
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Response {
        #[prost(message, tag="21")]
        CreateSessionResponse(super::CreateSessionResponse),
        #[prost(message, tag="22")]
        ExtendSessionResponse(super::ExtendSessionResponse),
        #[prost(message, tag="23")]
        PartialRunSetupResponse(super::PartialRunSetupResponse),
        #[prost(message, tag="24")]
        RunStepResponse(super::RunStepResponse),
        #[prost(message, tag="25")]
        CloseSessionResponse(super::CloseSessionResponse),
        #[prost(message, tag="26")]
        ListDevicesResponse(super::ListDevicesResponse),
        #[prost(message, tag="27")]
        ResetRequestResponse(super::ResetResponse),
        #[prost(message, tag="28")]
        MakeCallableResponse(super::MakeCallableResponse),
        #[prost(message, tag="29")]
        RunCallableResponse(super::RunCallableResponse),
        #[prost(message, tag="30")]
        ReleaseCallableResponse(super::ReleaseCallableResponse),
    }
}
/// A convolution. Currently it's only used for logging. In the future, we may
/// want to use it in the API as well.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConvolutionProto {
    #[prost(enumeration="super::stream_executor::dnn::ConvolutionKind", tag="1")]
    pub kind: i32,
    #[prost(message, optional, tag="2")]
    pub input: ::std::option::Option<super::stream_executor::dnn::TensorDescriptorProto>,
    #[prost(message, optional, tag="3")]
    pub filter: ::std::option::Option<super::stream_executor::dnn::TensorDescriptorProto>,
    #[prost(message, optional, tag="4")]
    pub output: ::std::option::Option<super::stream_executor::dnn::TensorDescriptorProto>,
    #[prost(message, optional, tag="5")]
    pub conv_desc: ::std::option::Option<super::stream_executor::dnn::ConvolutionDescriptorProto>,
    /// result = conv_scale * conv(...) + side_value_scale * side_value.
    /// side_value is an arbitrary buffer if activation is not none. Otherwise, it
    /// has to be the result buffer (using its old values).
    #[prost(double, tag="6")]
    pub conv_scale: f64,
    #[prost(double, tag="7")]
    pub side_value_scale: f64,
    #[prost(enumeration="super::stream_executor::dnn::ActivationMode", tag="8")]
    pub activation: i32,
    #[prost(int64, tag="9")]
    pub input_address: i64,
    #[prost(int64, tag="10")]
    pub filter_address: i64,
    #[prost(int64, tag="11")]
    pub output_address: i64,
    #[prost(int64, tag="12")]
    pub bias_address: i64,
    #[prost(int64, tag="13")]
    pub side_input_address: i64,
}
////////////////////////////////////////////////////////////////////////////////
//
// GetStatus method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetStatusRequest {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetStatusResponse {
    #[prost(message, repeated, tag="1")]
    pub device_attributes: ::std::vec::Vec<DeviceAttributes>,
}
////////////////////////////////////////////////////////////////////////////////
//
// CreateSession method request/response messages
//
// For each session,
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateWorkerSessionRequest {
    /// Sessions are identified by a given handle.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// Defines the configuration of a TensorFlow worker.
    #[prost(message, optional, tag="2")]
    pub server_def: ::std::option::Option<ServerDef>,
    /// If true, any resources such as Variables used in the session will not be
    /// shared with other sessions.
    #[prost(bool, tag="3")]
    pub isolate_session_state: bool,
    /// The device attributes of all the devices in the cluster.
    #[prost(message, repeated, tag="4")]
    pub cluster_device_attributes: ::std::vec::Vec<DeviceAttributes>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateWorkerSessionResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// DeleteSession method request/response messages
//
// Deletes all worker-side state associated with the given session handle.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteWorkerSessionRequest {
    /// Sessions are identified by a given handle.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteWorkerSessionResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// RegisterGraph method request/response messages
//
// For each session, after the master placed every node on a device,
// it partitions the whole graph into many subgraphs. All the nodes in
// a subgraph were in the same worker, but potentially on many devices
// owned by that worker (e.g. cpu0, plus gpu0, gpu1, ..., gpu7). The
// master registers subgraphs for a worker before running any steps. A
// successful registration returns a graph handle to be used in latter
// RunGraph requests.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterGraphRequest {
    /// Subgraphs are scoped within one session.
    #[prost(string, tag="1")]
    pub session_handle: std::string::String,
    /// Set to true if `CreateWorkerSession` was called for `session_handle`.
    #[prost(bool, tag="6")]
    pub create_worker_session_called: bool,
    /// "graph_def" has the subgraph of nodes for this worker, with each node
    /// having its device_name filled in.
    #[prost(message, optional, tag="2")]
    pub graph_def: ::std::option::Option<GraphDef>,
    /// True iff the graph (before partitioning) contains control flow nodes.
    ///
    /// As of 01/11/2015, this is no longer set by clients.
    #[prost(bool, tag="3")]
    pub has_control_flow: bool,
    /// Configuration options for the session in which this graph was created.
    #[prost(message, optional, tag="4")]
    pub graph_options: ::std::option::Option<GraphOptions>,
    /// Field(s) used by TensorFlow Debugger (tfdbg).
    #[prost(message, optional, tag="5")]
    pub debug_options: ::std::option::Option<DebugOptions>,
    /// If graph_def contains any collective ops this must be a positive
    /// integer used to coordinate execution with other graphs.  All
    /// graphs in a distributed execution with the same
    /// collective_graph_key will coordinate to use the same step_id
    /// concurrently so that BufRendezvous entries will make the correct
    /// values accessible.
    #[prost(int64, tag="7")]
    pub collective_graph_key: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterGraphResponse {
    /// If the registration succeeds, returns an opaque graph_handle to
    /// the master. The master calls RunGraph with graph_handle to
    /// compute different steps.
    #[prost(string, tag="1")]
    pub graph_handle: std::string::String,
}
////////////////////////////////////////////////////////////////////////////////
//
// DeregisterGraph method request/response messages
//
// The master deregisters the given graph_handle when the graph is no
// longer needed (e.g., the overall graph is re-scheduled and nodes
// are re-placed).
//
// The worker deregisters a graph_handle automatically according to on
// a TTL-base policy in case of master restarts.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeregisterGraphRequest {
    /// The session_handle used when registering the graph. If session_handle is
    /// empty, a single global namespace is used.
    #[prost(string, tag="2")]
    pub session_handle: std::string::String,
    /// Set to true if `CreateWorkerSession` was called for `session_handle`.
    #[prost(bool, tag="3")]
    pub create_worker_session_called: bool,
    /// REQUIRED: graph_handle must be returned by a RegisterGraph call
    /// to the same WorkerService.
    #[prost(string, tag="1")]
    pub graph_handle: std::string::String,
}
/// TODO(mrry): Optionally add summary stats for the graph.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeregisterGraphResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// CleanupAll method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CleanupAllRequest {
    /// A list of container names.
    ///
    /// If 'container' is not empty, releases resources in the given
    /// containers in all devices.
    ///
    /// If 'container' is empty, releases resources in the default
    /// container in all devices.
    #[prost(string, repeated, tag="1")]
    pub container: ::std::vec::Vec<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CleanupAllResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// RunGraph request / response messages
//
// The worker executes all subgraphs registered under graph_handle.
// RunGraph returns after the execution finishes or an error is
// encountered.
// A sequence of RunGraphRequests with is_partial may be sent to RunGraph for
// partial graph execution.
//
////////////////////////////////////////////////////////////////////////////////

/// Options specific to the execution of a single step.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutorOpts {
    #[prost(bool, tag="1")]
    pub record_costs: bool,
    #[prost(bool, tag="3")]
    pub record_timeline: bool,
    #[prost(bool, tag="4")]
    pub record_partition_graphs: bool,
    #[prost(bool, tag="5")]
    pub report_tensor_allocations_upon_oom: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunGraphRequest {
    /// session_handle is the master-generated unique id for this session.
    /// If session_handle is non-empty, it must be the same as used when
    /// registering the graph. If it is empty, a single global namespace is used to
    /// search for the graph_handle.
    #[prost(string, tag="8")]
    pub session_handle: std::string::String,
    /// Set to true if `CreateWorkerSession` was called for `session_handle`.
    #[prost(bool, tag="10")]
    pub create_worker_session_called: bool,
    /// REQUIRED: graph_handle must be returned by a RegisterGraph call
    /// to the same WorkerService.
    #[prost(string, tag="1")]
    pub graph_handle: std::string::String,
    /// A unique ID to distinguish different runs of the same graph.
    ///
    /// The master generates a global unique `step_id` to distinguish
    /// different runs of the graph computation. Subgraphs communicate
    /// (e.g., send/recv ops) with each other using `step_id` to
    /// distinguish tensors generated by different runs.
    #[prost(int64, tag="2")]
    pub step_id: i64,
    /// Options for this step.
    #[prost(message, optional, tag="5")]
    pub exec_opts: ::std::option::Option<ExecutorOpts>,
    /// Runs the graph.
    ///
    /// Sends the tensors in "send" into the graph before the run and
    /// fetches the keys into `RunGraphResponse.recv` after the run.
    #[prost(message, repeated, tag="3")]
    pub send: ::std::vec::Vec<NamedTensorProto>,
    #[prost(string, repeated, tag="4")]
    pub recv_key: ::std::vec::Vec<std::string::String>,
    /// True if the RunGraphRequest is a partial run request.
    #[prost(bool, tag="6")]
    pub is_partial: bool,
    /// True if this is the last partial run request in a sequence of requests.
    #[prost(bool, tag="7")]
    pub is_last_partial_run: bool,
    /// If true then some errors, e.g., execution errors that have long
    /// error messages, may return an OK RunGraphResponse with the actual
    /// error saved in the status_code/status_error_message fields of the
    /// response body. This is a workaround since the RPC subsystem may
    /// truncate long metadata messages.
    #[prost(bool, tag="9")]
    pub store_errors_in_response_body: bool,
    /// Unique identifier for this request. Every RunGraphRequest must have a
    /// unique request_id, and retried RunGraphRequests must have the same
    /// request_id. If request_id is zero, retry detection is disabled.
    ///
    /// Retried RunGraphRequests are problematic because they may issue a
    /// RecvTensor that will have no corresponding sender and will wait forever.
    /// Workers use request_ids to reject retried RunGraph requests instead of
    /// waiting forever.
    #[prost(int64, tag="11")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunGraphResponse {
    /// A list of tensors corresponding to those requested by
    /// `RunGraphRequest.recv_key`.
    #[prost(message, repeated, tag="1")]
    pub recv: ::std::vec::Vec<NamedTensorProto>,
    /// If the request asked for execution stats, the cost graph, or the partition
    /// graphs, these are returned here.
    /// TODO(suharshs): Package these in a RunMetadata instead.
    #[prost(message, optional, tag="2")]
    pub step_stats: ::std::option::Option<StepStats>,
    #[prost(message, optional, tag="3")]
    pub cost_graph: ::std::option::Option<CostGraphDef>,
    #[prost(message, repeated, tag="4")]
    pub partition_graph: ::std::vec::Vec<GraphDef>,
    /// If store_errors_in_response_body is true in the request, then
    /// optionally the server may return an OK status for the RPC and
    /// fill the true status into the fields below, to allow for messages
    /// that are too long to fit in metadata.
    #[prost(enumeration="error::Code", tag="5")]
    pub status_code: i32,
    #[prost(string, tag="6")]
    pub status_error_message: std::string::String,
}
////////////////////////////////////////////////////////////////////////////////
//
// CleanupGraph method request/response messages
//
// After the master receives RunGraph responses from all workers, the
// master instructs every worker to cleanup any remaining state of a
// step (e.g. tensors buffered by a `Send` op but not picked up by
// other workers). The master does not necessarily need to wait for
// completion of CleanupGraph calls.
//
// Workers should cleanup step states automatically according to a
// TTL-based policy in case of master restarts.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CleanupGraphRequest {
    #[prost(int64, tag="1")]
    pub step_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CleanupGraphResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// RecvTensor method request/response messages
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecvTensorRequest {
    /// The step in which the tensor will be produced.
    ///
    /// REQUIRED: This must eventually correspond to the `step_id` passed
    /// into a RunGraph call on the same WorkerService.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// A key identifying the channel to receive tensors from. A RecvTensor request
    /// retrieves one tensor from the channel, but multiple tensors can be sent and
    /// received over the same channel with multiple RecvTensor requests. See
    /// rendezvous.h for details.
    #[prost(string, tag="2")]
    pub rendezvous_key: std::string::String,
    /// If true, use an out-of-band DMA mechanism to transfer the
    /// received tensor.
    #[prost(bool, tag="3")]
    pub dma_ok: bool,
    /// Optional information on client-side device locality.
    #[prost(message, optional, tag="4")]
    pub client_locality: ::std::option::Option<DeviceLocality>,
    /// Optional information on server-side device locality.
    #[prost(message, optional, tag="5")]
    pub server_locality: ::std::option::Option<DeviceLocality>,
    /// Optional information needed by the RPC subsystem.
    #[prost(message, optional, tag="6")]
    pub transport_options: ::std::option::Option<::prost_types::Any>,
    /// Unique identifier for this request. Every RecvTensorRequest must have a
    /// unique request_id, and retried RecvTensorRequests must have the same
    /// request_id. If request_id is zero, retry detection and response cache
    /// are disabled.
    ///
    /// Retried RecvTensorRequests are problematic because a RecvTensor with no
    /// corresponding sender will wait forever, and the tensor may have been
    /// delivered to a previous retry. Workers use request_ids to reject retried
    /// RecvTensor requests instead of waiting forever.
    #[prost(int64, tag="7")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecvTensorResponse {
    /// The tensor as a proto.
    #[prost(message, optional, tag="1")]
    pub tensor: ::std::option::Option<TensorProto>,
    /// If true, this tensor was the output of a dead node, and the
    /// content is invalid.
    #[prost(bool, tag="2")]
    pub is_dead: bool,
    /// The time at which tensor was available and started to be returned.
    #[prost(int64, tag="3")]
    pub send_start_micros: i64,
    /// Optional additional information about how to receive the tensor,
    /// e.g. in the event that `RecvTensorRequest.dma_ok` was true.
    #[prost(message, optional, tag="4")]
    pub transport_options: ::std::option::Option<::prost_types::Any>,
    /// Whether the receiver should send a MarkRecvFinishedRequest to the sender
    /// to ack the message.
    #[prost(bool, tag="5")]
    pub require_ack: bool,
}
/// Message for managing the response cache maintained on the sender side.
/// Currently only used by the gRPC worker service.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MarkRecvFinishedRequest {
    #[prost(int64, tag="1")]
    pub request_id: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MarkRecvFinishedResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// Logging method request/response messages
//
// NOTE(mrry): This feature is not supported in the open-source
// version, and these messages are expected to change.
//
////////////////////////////////////////////////////////////////////////////////

/// Out-of-band request to begin or end logging, or
/// to retrieve logs for particular steps.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LoggingRequest {
    /// If true, RPC logging will be enabled.
    #[prost(bool, tag="1")]
    pub enable_rpc_logging: bool,
    /// If true, RPC logging will be disabled.
    #[prost(bool, tag="4")]
    pub disable_rpc_logging: bool,
    /// If true, discard any saved logging data (for all steps).
    #[prost(bool, tag="2")]
    pub clear: bool,
    /// When set, requests all saved log data pertaining to the step.
    /// Any log data retrieved is eliminated from the store and cannot be
    /// retrieved again.
    #[prost(int64, repeated, tag="3")]
    pub fetch_step_id: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LabeledStepStats {
    #[prost(int64, tag="1")]
    pub step_id: i64,
    #[prost(message, optional, tag="2")]
    pub step_stats: ::std::option::Option<StepStats>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LoggingResponse {
    #[prost(message, repeated, tag="1")]
    pub step: ::std::vec::Vec<LabeledStepStats>,
}
////////////////////////////////////////////////////////////////////////////////
//
// Tracing method request/response messages
//
// NOTE(mrry): This feature is not supported in the open-source
// version, and these messages are expected to change.
//
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TraceOpts {
    /// Length of the trace to be taken, in seconds.
    #[prost(double, tag="1")]
    pub duration: f64,
    /// If true, capture step profile locally in each worker. Currently
    /// unimplemented.
    #[prost(bool, tag="2")]
    pub use_step_profiler: bool,
    /// If true, capture kernel events from each worker.
    #[prost(bool, tag="3")]
    pub use_kernel_profiler: bool,
    /// If true, capture extended profiling events from TensorFlow process.
    #[prost(bool, tag="4")]
    pub use_extended_profiler: bool,
    /// If true, capture GPU profiling events locally on each
    /// machine. Currently unimplemented.
    #[prost(bool, tag="5")]
    pub use_gpu_profiler: bool,
    /// If true, collect sampled profile events. Currently unimplemented.
    #[prost(bool, tag="6")]
    pub use_sample_profiler: bool,
}
/// Out-of-band request to configure distributed tracing.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TracingRequest {
    #[prost(message, optional, tag="1")]
    pub options: ::std::option::Option<TraceOpts>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TracingResponse {
}
////////////////////////////////////////////////////////////////////////////////
//
// Raw data transfers in support of Collective Ops.
// These methods are experimental and subject to change.
//
// The intention is to allow collectives to take advantage of the most
// efficient methods available on a platform, e.g. RDMA, and not be
// constrained to use the RPC system in use by other methods.
//
////////////////////////////////////////////////////////////////////////////////

/// Use of the fields below may vary by implementation.  For example
/// the buf_ptr and num_bytes may be set only for local operations and
/// not sent on the wire, or only sent on the wire in one direction.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecvBufRequest {
    /// Used at server side to find the correct BufRendezvous.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Arbitrary string identifying a BufRendezvous entry.
    #[prost(string, tag="2")]
    pub buf_rendezvous_key: std::string::String,
    /// Size of value expected, must agree with BufRendezvous entry.
    #[prost(int64, tag="3")]
    pub num_bytes: i64,
    /// When RDMA is in use, address of destination field on client.
    #[prost(fixed64, tag="4")]
    pub buf_ptr: u64,
    /// Optional information on client-side device locality.
    #[prost(message, optional, tag="5")]
    pub client_locality: ::std::option::Option<DeviceLocality>,
    /// Optional information on server-side device locality.
    #[prost(message, optional, tag="6")]
    pub server_locality: ::std::option::Option<DeviceLocality>,
    /// Optional, implementation-specific data.
    #[prost(message, optional, tag="7")]
    pub transport_options: ::std::option::Option<::prost_types::Any>,
    /// For annotating timeline and device incarnation check.
    #[prost(string, tag="8")]
    pub src_device: std::string::String,
    /// Optional, for annotating the timeline.
    #[prost(string, tag="9")]
    pub dst_device: std::string::String,
    /// Depending on the RPC system in use, it may be necessary to set this
    /// id to detect resends of RPCs where the server is not aware that
    /// the prior RPC failed.
    #[prost(int64, tag="10")]
    pub request_id: i64,
    /// Incarnation number of the source device, used to detect worker failures.
    #[prost(uint64, tag="11")]
    pub src_incarnation: u64,
}
/// Use of the fields below may vary by implementation.  Comments give
/// intended use.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecvBufResponse {
    /// Address of source field on server.
    #[prost(fixed64, tag="1")]
    pub buf_ptr: u64,
    /// Byte length of buf_ptr field, if set.
    #[prost(int64, tag="2")]
    pub num_bytes: i64,
    /// True if value is 'dead' like a tensor.
    #[prost(bool, tag="3")]
    pub is_dead: bool,
    /// Optional, implementation-specific data.
    #[prost(message, optional, tag="4")]
    pub transport_options: ::std::option::Option<::prost_types::Any>,
    /// Optional, for timeline.
    #[prost(int64, tag="5")]
    pub send_start_micros: i64,
    /// Whether the receiver should send a MarkRecvFinishedRequest to the sender
    /// to ack the message.
    #[prost(bool, tag="6")]
    pub require_ack: bool,
}
////////////////////////////////////////////////////////////////////////////////
//
// Collective Op dynamic group resolution messages.
//
////////////////////////////////////////////////////////////////////////////////

/// Supplies one or more device names as members of the group identified by
/// group_key.  Service will respond when all group_size devices become known.
/// All devices in group must have same type.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompleteGroupRequest {
    #[prost(int32, tag="1")]
    pub group_key: i32,
    #[prost(int32, tag="2")]
    pub group_size: i32,
    #[prost(string, tag="3")]
    pub device_type: std::string::String,
    #[prost(string, repeated, tag="4")]
    pub device_name: ::std::vec::Vec<std::string::String>,
    #[prost(int32, tag="5")]
    pub collective_type: i32,
}
/// Gives the complete membership of the group identified by group_key.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompleteGroupResponse {
    #[prost(int32, tag="1")]
    pub group_key: i32,
    #[prost(int32, tag="2")]
    pub group_size: i32,
    #[prost(string, tag="3")]
    pub device_type: std::string::String,
    /// number of distinct tasks hosting the devices
    #[prost(int32, tag="4")]
    pub num_tasks: i32,
    #[prost(string, repeated, tag="5")]
    pub device_name: ::std::vec::Vec<std::string::String>,
    /// task name prefixes of device_names
    #[prost(string, repeated, tag="6")]
    pub task_name: ::std::vec::Vec<std::string::String>,
    #[prost(bytes, tag="7")]
    pub communicator_key: std::vec::Vec<u8>,
}
/// Supplies data about one collective op belonging to the instance identified
/// by instance_key.  Service will respond when all group_size ops have
/// become known.  Most of the data being sent is for correctness checking,
/// to ensure that all ops in the instance share common attributes.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompleteInstanceRequest {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(int32, tag="2")]
    pub r#type: i32,
    #[prost(enumeration="DataType", tag="3")]
    pub data_type: i32,
    #[prost(message, optional, tag="4")]
    pub shape: ::std::option::Option<TensorShapeProto>,
    #[prost(int32, tag="5")]
    pub group_key: i32,
    #[prost(int32, tag="6")]
    pub group_size: i32,
    #[prost(int32, tag="7")]
    pub instance_key: i32,
    #[prost(string, tag="8")]
    pub device_type: std::string::String,
    #[prost(int32, repeated, tag="9")]
    pub subdiv_offset: ::std::vec::Vec<i32>,
    #[prost(string, tag="10")]
    pub device: std::string::String,
    #[prost(bool, tag="11")]
    pub is_source: bool,
}
/// Confirms that every op in the instance has consistently declared itself.
/// Also gives the source_rank in case of broadcast.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompleteInstanceResponse {
    #[prost(int32, tag="1")]
    pub instance_key: i32,
    #[prost(int32, tag="2")]
    pub source_rank: i32,
}
/// Request for next agreed-upon step_id for the specified graph_keys.
/// This is used to enable multiple graphs containing nodes from
/// a common collective instance to coordinate using the same step_ids.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetStepSequenceRequest {
    #[prost(int64, repeated, tag="1")]
    pub graph_key: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StepSequence {
    #[prost(int64, tag="1")]
    pub graph_key: i64,
    #[prost(int64, tag="2")]
    pub next_step_id: i64,
}
/// Next valid step_ids for one or more graph_keys.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetStepSequenceResponse {
    #[prost(message, repeated, tag="1")]
    pub step_sequence: ::std::vec::Vec<StepSequence>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceProperties {
    /// Device type (CPU, GPU, ...)
    #[prost(string, tag="1")]
    pub r#type: std::string::String,
    /// Vendor (Intel, nvidia, ...)
    #[prost(string, tag="2")]
    pub vendor: std::string::String,
    /// Model (Haswell, K40, ...)
    #[prost(string, tag="3")]
    pub model: std::string::String,
    /// Core Frequency in Mhz
    #[prost(int64, tag="4")]
    pub frequency: i64,
    /// Number of cores
    #[prost(int64, tag="5")]
    pub num_cores: i64,
    /// Version of the tools and libraries used with this device (e.g. gcc 4.9,
    /// cudnn 5.1)
    #[prost(map="string, string", tag="6")]
    pub environment: ::std::collections::HashMap<std::string::String, std::string::String>,
    /// Number of registers per core.
    #[prost(int64, tag="7")]
    pub num_registers: i64,
    /// L1 cache size in bytes
    #[prost(int64, tag="8")]
    pub l1_cache_size: i64,
    /// L2 cache size in bytes
    #[prost(int64, tag="9")]
    pub l2_cache_size: i64,
    /// L3 cache size in bytes
    #[prost(int64, tag="10")]
    pub l3_cache_size: i64,
    /// Shared memory size per multiprocessor in bytes. This field is
    /// applicable to GPUs only.
    #[prost(int64, tag="11")]
    pub shared_memory_size_per_multiprocessor: i64,
    /// Memory size in bytes
    #[prost(int64, tag="12")]
    pub memory_size: i64,
    /// Memory bandwidth in KB/s
    #[prost(int64, tag="13")]
    pub bandwidth: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NamedDevice {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(message, optional, tag="2")]
    pub properties: ::std::option::Option<DeviceProperties>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferNodeInput {
    #[prost(int32, tag="1")]
    pub node_id: i32,
    #[prost(int32, tag="2")]
    pub output_port: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferNodeInfo {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(int32, tag="2")]
    pub node_id: i32,
    #[prost(string, tag="3")]
    pub type_name: std::string::String,
    #[prost(int32, tag="4")]
    pub soc_op_id: i32,
    #[prost(int32, tag="5")]
    pub padding_id: i32,
    #[prost(int32, tag="6")]
    pub input_count: i32,
    #[prost(int32, tag="7")]
    pub output_count: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferConstNodeInfo {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(int32, tag="2")]
    pub node_id: i32,
    #[prost(int64, repeated, tag="3")]
    pub shape: ::std::vec::Vec<i64>,
    #[prost(bytes, tag="4")]
    pub data: std::vec::Vec<u8>,
    #[prost(enumeration="DataType", tag="5")]
    pub dtype: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferNodeInputInfo {
    #[prost(int32, tag="1")]
    pub node_id: i32,
    #[prost(message, repeated, tag="2")]
    pub node_input: ::std::vec::Vec<GraphTransferNodeInput>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferNodeOutputInfo {
    #[prost(int32, tag="1")]
    pub node_id: i32,
    #[prost(int32, repeated, tag="2")]
    pub max_byte_size: ::std::vec::Vec<i32>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferGraphInputNodeInfo {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(int64, repeated, tag="2")]
    pub shape: ::std::vec::Vec<i64>,
    #[prost(enumeration="DataType", tag="3")]
    pub dtype: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferGraphOutputNodeInfo {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(int64, repeated, tag="2")]
    pub shape: ::std::vec::Vec<i64>,
    #[prost(enumeration="DataType", tag="3")]
    pub dtype: i32,
}
/// Protocol buffer representing a handle to a tensorflow resource. Handles are
/// not valid across executions, but can be serialized back and forth from within
/// a single run.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphTransferInfo {
    #[prost(message, repeated, tag="1")]
    pub node_info: ::std::vec::Vec<GraphTransferNodeInfo>,
    #[prost(message, repeated, tag="2")]
    pub const_node_info: ::std::vec::Vec<GraphTransferConstNodeInfo>,
    #[prost(message, repeated, tag="3")]
    pub node_input_info: ::std::vec::Vec<GraphTransferNodeInputInfo>,
    #[prost(message, repeated, tag="4")]
    pub node_output_info: ::std::vec::Vec<GraphTransferNodeOutputInfo>,
    /// Input Node parameters of transferred graph
    #[prost(message, repeated, tag="5")]
    pub graph_input_node_info: ::std::vec::Vec<GraphTransferGraphInputNodeInfo>,
    #[prost(message, repeated, tag="6")]
    pub graph_output_node_info: ::std::vec::Vec<GraphTransferGraphOutputNodeInfo>,
    /// Destination of graph transfer
    #[prost(enumeration="graph_transfer_info::Destination", tag="7")]
    pub destination: i32,
}
pub mod graph_transfer_info {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Destination {
        Nop = 0,
        Hexagon = 1,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KernelDef {
    /// Must match the name of an Op.
    #[prost(string, tag="1")]
    pub op: std::string::String,
    /// Type of device this kernel runs on.
    #[prost(string, tag="2")]
    pub device_type: std::string::String,
    #[prost(message, repeated, tag="3")]
    pub constraint: ::std::vec::Vec<kernel_def::AttrConstraint>,
    /// Names of the Op's input_/output_args that reside in host memory
    /// instead of device memory.
    #[prost(string, repeated, tag="4")]
    pub host_memory_arg: ::std::vec::Vec<std::string::String>,
    /// This allows experimental kernels to be registered for an op that
    /// won't be used unless the user specifies a "_kernel" attr with
    /// value matching this.
    #[prost(string, tag="5")]
    pub label: std::string::String,
    /// Prioritization of kernel amongst different devices. By default we assume
    /// priority is 0. The higher the priority the better. By default (i.e. if
    /// this is not set), we prefer GPU kernels over CPU.
    #[prost(int32, tag="6")]
    pub priority: i32,
}
pub mod kernel_def {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AttrConstraint {
        /// Name of an attr from the Op.
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// A list of values that this kernel supports for this attr.
        /// Like OpDef.AttrDef.allowed_values, except for kernels instead of Ops.
        #[prost(message, optional, tag="2")]
        pub allowed_values: ::std::option::Option<super::AttrValue>,
    }
}
/// A collection of KernelDefs
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KernelList {
    #[prost(message, repeated, tag="1")]
    pub kernel: ::std::vec::Vec<KernelDef>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogStep {
    /// Process-unique step id.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Handle describing the feeds and fetches of the step.
    #[prost(string, tag="2")]
    pub handle: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogTensorAllocation {
    /// Process-unique step id.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Name of the kernel making the allocation as set in GraphDef,
    /// e.g., "affine2/weights/Assign".
    #[prost(string, tag="2")]
    pub kernel_name: std::string::String,
    /// Allocated tensor details.
    #[prost(message, optional, tag="3")]
    pub tensor: ::std::option::Option<TensorDescription>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogTensorDeallocation {
    /// Id of the tensor buffer being deallocated, used to match to a
    /// corresponding allocation.
    #[prost(int64, tag="1")]
    pub allocation_id: i64,
    /// Name of the allocator used.
    #[prost(string, tag="2")]
    pub allocator_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogTensorOutput {
    /// Process-unique step id.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Name of the kernel producing an output as set in GraphDef, e.g.,
    /// "affine2/weights/Assign".
    #[prost(string, tag="2")]
    pub kernel_name: std::string::String,
    /// Index of the output being set.
    #[prost(int32, tag="3")]
    pub index: i32,
    /// Output tensor details.
    #[prost(message, optional, tag="4")]
    pub tensor: ::std::option::Option<TensorDescription>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogRawAllocation {
    /// Process-unique step id.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Name of the operation making the allocation.
    #[prost(string, tag="2")]
    pub operation: std::string::String,
    /// Number of bytes in the allocation.
    #[prost(int64, tag="3")]
    pub num_bytes: i64,
    /// Address of the allocation.
    #[prost(uint64, tag="4")]
    pub ptr: u64,
    /// Id of the tensor buffer being allocated, used to match to a
    /// corresponding deallocation.
    #[prost(int64, tag="5")]
    pub allocation_id: i64,
    /// Name of the allocator used.
    #[prost(string, tag="6")]
    pub allocator_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryLogRawDeallocation {
    /// Process-unique step id.
    #[prost(int64, tag="1")]
    pub step_id: i64,
    /// Name of the operation making the deallocation.
    #[prost(string, tag="2")]
    pub operation: std::string::String,
    /// Id of the tensor buffer being deallocated, used to match to a
    /// corresponding allocation.
    #[prost(int64, tag="3")]
    pub allocation_id: i64,
    /// Name of the allocator used.
    #[prost(string, tag="4")]
    pub allocator_name: std::string::String,
    /// True if the deallocation is queued and will be performed later,
    /// e.g. for GPU lazy freeing of buffers.
    #[prost(bool, tag="5")]
    pub deferred: bool,
}
/// For serializing and restoring the state of ReaderBase, see
/// reader_base.h for details.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReaderBaseState {
    #[prost(int64, tag="1")]
    pub work_started: i64,
    #[prost(int64, tag="2")]
    pub work_finished: i64,
    #[prost(int64, tag="3")]
    pub num_records_produced: i64,
    #[prost(bytes, tag="4")]
    pub current_work: std::vec::Vec<u8>,
}
/// Used to specify and override the default API & behavior in the
/// generated code for client languages, from what you would get from
/// the OpDef alone. There will be a set of ApiDefs that are common
/// to all client languages, and another set per client language.
/// The per-client-language ApiDefs will inherit values from the
/// common ApiDefs which it can either replace or modify.
///
/// We separate the API definition from the OpDef so we can evolve the
/// API while remaining backwards compatible when interpretting old
/// graphs.  Overrides go in an "api_def.pbtxt" file with a text-format
/// ApiDefs message.
///
/// WARNING: Be *very* careful changing the API for any existing op --
/// you can change the semantics of existing code.  These changes may
/// need to wait until a major release of TensorFlow to avoid breaking
/// our compatibility promises.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApiDef {
    /// Name of the op (in the OpDef) to specify the API for.
    #[prost(string, tag="1")]
    pub graph_op_name: std::string::String,
    /// If this op is deprecated, set deprecation message to the message
    /// that should be logged when this op is used.
    /// The message should indicate alternative op to use, if any.
    #[prost(string, tag="12")]
    pub deprecation_message: std::string::String,
    /// Major version when the op will be deleted. For e.g. set this
    /// value to 2 if op API should be removed in TensorFlow 2.0 and
    /// deprecated in versions before that.
    #[prost(int32, tag="13")]
    pub deprecation_version: i32,
    #[prost(enumeration="api_def::Visibility", tag="2")]
    pub visibility: i32,
    #[prost(message, repeated, tag="3")]
    pub endpoint: ::std::vec::Vec<api_def::Endpoint>,
    #[prost(message, repeated, tag="4")]
    pub in_arg: ::std::vec::Vec<api_def::Arg>,
    #[prost(message, repeated, tag="5")]
    pub out_arg: ::std::vec::Vec<api_def::Arg>,
    /// List of original in_arg names to specify new argument order.
    /// Length of arg_order should be either empty to keep current order
    /// or match size of in_arg.
    #[prost(string, repeated, tag="11")]
    pub arg_order: ::std::vec::Vec<std::string::String>,
    #[prost(message, repeated, tag="6")]
    pub attr: ::std::vec::Vec<api_def::Attr>,
    /// One-line human-readable description of what the Op does.
    #[prost(string, tag="7")]
    pub summary: std::string::String,
    /// Additional, longer human-readable description of what the Op does.
    #[prost(string, tag="8")]
    pub description: std::string::String,
    /// Modify an existing/inherited description by adding text to the beginning
    /// or end.
    #[prost(string, tag="9")]
    pub description_prefix: std::string::String,
    #[prost(string, tag="10")]
    pub description_suffix: std::string::String,
}
pub mod api_def {
    /// If you specify any endpoint, this will replace all of the
    /// inherited endpoints.  The first endpoint should be the
    /// "canonical" endpoint, and should not be deprecated (unless all
    /// endpoints are deprecated).
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Endpoint {
        /// Name should be either like "CamelCaseName" or
        /// "Package.CamelCaseName". Client-language-specific ApiDefs may
        /// use a snake_case convention instead of CamelCase.
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// Set if this endpoint is deprecated. If set to true, a message suggesting
        /// to use a non-deprecated endpoint instead will be printed. If all
        /// endpoints are deprecated, set deprecation_message in ApiDef instead.
        #[prost(bool, tag="3")]
        pub deprecated: bool,
        /// Major version when an endpoint will be deleted. For e.g. set this
        /// value to 2 if endpoint should be removed in TensorFlow 2.0 and
        /// deprecated in versions before that.
        #[prost(int32, tag="4")]
        pub deprecation_version: i32,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Arg {
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// Change the name used to access this arg in the API from what
        /// is used in the GraphDef.  Note that these names in `backticks`
        /// will also be replaced in the summary & description fields.
        #[prost(string, tag="2")]
        pub rename_to: std::string::String,
        /// Note: this will replace any inherited arg doc. There is no
        /// current way of modifying arg descriptions (other than replacing
        /// them entirely) as can be done with op descriptions.
        #[prost(string, tag="3")]
        pub description: std::string::String,
    }
    /// Description of the graph-construction-time configuration of this
    /// Op.  That is to say, this describes the attr fields that will
    /// be specified in the NodeDef.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Attr {
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// Change the name used to access this attr in the API from what
        /// is used in the GraphDef.  Note that these names in `backticks`
        /// will also be replaced in the summary & description fields.
        #[prost(string, tag="2")]
        pub rename_to: std::string::String,
        /// Specify a new default value to use for this attr.  This default
        /// will be used when creating new graphs, as opposed to the
        /// default in the OpDef, which will be used when interpreting old
        /// GraphDefs.
        #[prost(message, optional, tag="3")]
        pub default_value: ::std::option::Option<super::AttrValue>,
        /// Note: this will replace any inherited attr doc, there is no current
        /// way of modifying attr descriptions as can be done with op descriptions.
        #[prost(string, tag="4")]
        pub description: std::string::String,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Visibility {
        /// Normally this is "VISIBLE" unless you are inheriting a
        /// different value from another ApiDef.
        DefaultVisibility = 0,
        /// Publicly visible in the API.
        Visible = 1,
        /// Do not include this op in the generated API. If visibility is
        /// set to 'SKIP', other fields are ignored for this op.
        Skip = 2,
        /// Hide this op by putting it into an internal namespace (or whatever
        /// is appropriate in the target language).
        Hidden = 3,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApiDefs {
    #[prost(message, repeated, tag="1")]
    pub op: ::std::vec::Vec<ApiDef>,
}
/// Protocol buffer representing a handle to a tensorflow resource. Handles are
/// not valid across executions, but can be serialized back and forth from within
/// a single run.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoteFusedGraphExecuteInfo {
    /// Definition of remote graph
    #[prost(message, optional, tag="1")]
    pub remote_graph: ::std::option::Option<GraphDef>,
    /// Remote fused graph input node name
    #[prost(string, repeated, tag="2")]
    pub graph_input_node_name: ::std::vec::Vec<std::string::String>,
    /// Remote fused graph output node name
    #[prost(string, repeated, tag="3")]
    pub graph_output_node_name: ::std::vec::Vec<std::string::String>,
    /// Executor's name
    #[prost(string, tag="4")]
    pub executor_name: std::string::String,
    /// Optional: Parameters given to the executor
    #[prost(bytes, tag="5")]
    pub serialized_executor_parameters: std::vec::Vec<u8>,
    /// Optional: Default graph input tensor shape used to allocate memory
    /// before executing op
    #[prost(message, repeated, tag="6")]
    pub default_graph_input_tensor_shape: ::std::vec::Vec<remote_fused_graph_execute_info::TensorShapeTypeProto>,
    /// Optional: Default graph input tensor shape used to allocate memory
    /// before executing op
    /// TODO(satok): Remote output tensor shape once shape information is stored
    /// in NodeDef
    #[prost(message, repeated, tag="7")]
    pub default_graph_output_tensor_shape: ::std::vec::Vec<remote_fused_graph_execute_info::TensorShapeTypeProto>,
}
pub mod remote_fused_graph_execute_info {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TensorShapeTypeProto {
        #[prost(enumeration="super::DataType", tag="1")]
        pub dtype: i32,
        #[prost(message, optional, tag="2")]
        pub shape: ::std::option::Option<super::TensorShapeProto>,
    }
}
/// Description of the session when an op is run.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SessionInfo {
    #[prost(int64, tag="1")]
    pub intra_op_parallelism: i64,
}
/// Description of an operation as well as the parameters expected to impact its
/// performance.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpInfo {
    /// The operation name.  There may be custom parameters in attrs.
    #[prost(string, tag="1")]
    pub op: std::string::String,
    /// Custom parameters impacting the behavior of the op.
    #[prost(map="string, message", tag="2")]
    pub attr: ::std::collections::HashMap<std::string::String, AttrValue>,
    #[prost(message, repeated, tag="3")]
    pub inputs: ::std::vec::Vec<op_info::TensorProperties>,
    /// Optional description of the op outputs
    #[prost(message, repeated, tag="5")]
    pub outputs: ::std::vec::Vec<op_info::TensorProperties>,
    /// Device on which the operation is run.
    #[prost(message, optional, tag="4")]
    pub device: ::std::option::Option<DeviceProperties>,
    /// Information about the session configs.
    #[prost(message, optional, tag="6")]
    pub session_info: ::std::option::Option<SessionInfo>,
}
pub mod op_info {
    /// Input data types, shapes and values if known.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct TensorProperties {
        #[prost(enumeration="super::DataType", tag="1")]
        pub dtype: i32,
        #[prost(message, optional, tag="2")]
        pub shape: ::std::option::Option<super::TensorShapeProto>,
        #[prost(message, optional, tag="3")]
        pub value: ::std::option::Option<super::TensorProto>,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NormalDistribution {
    #[prost(double, tag="1")]
    pub mu: f64,
    #[prost(double, tag="2")]
    pub sigma: f64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LogNormalDistribution {
    #[prost(double, tag="1")]
    pub mu: f64,
    #[prost(double, tag="2")]
    pub sigma: f64,
}
/// Performance data for tensorflow operations
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpPerformance {
    /// The op
    #[prost(message, optional, tag="1")]
    pub op: ::std::option::Option<OpInfo>,
    /// Information about the session configs.
    #[prost(message, optional, tag="12")]
    pub session_info: ::std::option::Option<SessionInfo>,
    /// The node name (optional). Makes it easier to associate the performance data
    /// with a specific graph node.
    #[prost(string, tag="5")]
    pub node: std::string::String,
    /// Temporary memory used by this node (in bytes).
    #[prost(int64, tag="2")]
    pub temporary_memory_size: i64,
    /// Time it takes to run the op (in nanoseconds).
    #[prost(int64, tag="3")]
    pub compute_cost: i64,
    /// Analytical compute cost (in nanoseconds).
    #[prost(int64, tag="6")]
    pub compute_time: i64,
    /// Analytical memory access cost (in nanoseconds).
    #[prost(int64, tag="7")]
    pub memory_time: i64,
    /// Percentage of theoretical compute performance.
    #[prost(double, tag="4")]
    pub compute_efficiency: f64,
    /// Percentage of theoretical memory performance.
    #[prost(double, tag="8")]
    pub memory_efficiency: f64,
    #[prost(message, optional, tag="9")]
    pub op_memory: ::std::option::Option<op_performance::OpMemory>,
    /// Expected execution time, modeled using one of 2 possible distributions.
    #[prost(oneof="op_performance::ExecutionTime", tags="10, 11")]
    pub execution_time: ::std::option::Option<op_performance::ExecutionTime>,
}
pub mod op_performance {
    /// Memory usage data for a tensorflow operation.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct OpMemory {
        /// The output information may have memory usage and output shapes.
        #[prost(int64, repeated, tag="1")]
        pub output_memory: ::std::vec::Vec<i64>,
        /// Temp and persistent memory allocated by this node.
        #[prost(int64, tag="2")]
        pub temp_memory: i64,
        #[prost(int64, tag="4")]
        pub persistent_memory: i64,
        #[prost(int64, tag="3")]
        pub device_temp_memory: i64,
        #[prost(int64, tag="5")]
        pub device_persistent_memory: i64,
    }
    /// Expected execution time, modeled using one of 2 possible distributions.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ExecutionTime {
        #[prost(message, tag="10")]
        ExecutionTimeNormal(super::NormalDistribution),
        #[prost(message, tag="11")]
        ExecutionTimeLogNormal(super::LogNormalDistribution),
    }
}
/// A collection of OpPerformance data points.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpPerformanceList {
    #[prost(message, repeated, tag="1")]
    pub op_performance: ::std::vec::Vec<OpPerformance>,
}
/// Summarizes the results of auto-clustering a TensorFlow graph.
///
/// Next ID: 5
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaAutoClusteringSummary {
    /// The number of nodes in the graph that are not inside an XLA cluster.
    #[prost(int32, tag="1")]
    pub unclustered_node_count: i32,
    /// The number of nodes in the graph that are in an XLA cluster.
    #[prost(int32, tag="2")]
    pub clustered_node_count: i32,
    /// All of the XLA clusters in the TF graph.
    #[prost(message, repeated, tag="3")]
    pub clusters: ::std::vec::Vec<xla_auto_clustering_summary::Cluster>,
    /// A histogram of the TF operations that were not clustered.
    #[prost(message, repeated, tag="4")]
    pub unclustered_op_histogram: ::std::vec::Vec<xla_auto_clustering_summary::OpAndCount>,
}
pub mod xla_auto_clustering_summary {
    /// Represents a single element in a histogram of ops ("op" as in "TensorFlow
    /// operation").
    ///
    /// Next ID: 3
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct OpAndCount {
        /// The TensorFlow operation (like MatMult, Add etc.)
        #[prost(string, tag="1")]
        pub op: std::string::String,
        /// The number of times this occurs.
        #[prost(int32, tag="2")]
        pub count: i32,
    }
    /// Describes a single XLA cluster.
    ///
    /// Next ID: 4
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Cluster {
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// The number of nodes in the cluster.
        #[prost(int32, tag="2")]
        pub size: i32,
        /// A histogram of the TF operations in this cluster.
        #[prost(message, repeated, tag="3")]
        pub op_histogram: ::std::vec::Vec<OpAndCount>,
    }
}
/// Listeners listening for auto clustering events get messages of this type.
///
/// Next ID: 4
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaAutoClusteringActivity {
    /// The value of GlobalJitLevel, as determined by `GetGlobalJitLevelForGraph`.
    /// This determines if global auto-clustering is enabled.
    #[prost(enumeration="optimizer_options::GlobalJitLevel", tag="1")]
    pub global_jit_level: i32,
    /// Whether --tf_xla_cpu_global_jit is enabled in TF_XLA_FLAGS.
    #[prost(bool, tag="2")]
    pub cpu_global_jit_enabled: bool,
    #[prost(message, optional, tag="3")]
    pub summary: ::std::option::Option<XlaAutoClusteringSummary>,
}
/// Listeners listening for JIT compilation events get messages of this type.
/// Each instance of XlaJitCompilationActivity corresponds to a single
/// compilation of a single XLA cluster.  E.g. if a graph has two clusters, A and
/// B, and A is compiled 5 times and B is compiled 2 times then we will generate
/// 7 instances of XlaJitCompilationActivity.
///
/// Next ID: 5
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaJitCompilationActivity {
    #[prost(string, tag="1")]
    pub cluster_name: std::string::String,
    /// The number of time this cluster has been compiled.
    #[prost(int32, tag="2")]
    pub compile_count: i32,
    /// Microseconds spent in the individual compilation being reported.
    #[prost(int64, tag="3")]
    pub compile_time_us: i64,
    /// Total microseconds spent in (re-)compiling this cluster so far.
    #[prost(int64, tag="4")]
    pub cumulative_compile_time_us: i64,
}
/// LINT.IfChange
///
/// Used for logging situations seen in Tensorflow models being optimized that
/// are known to not perform well with XLA.
///
/// Next ID: 3
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaOptimizationRemark {
    #[prost(enumeration="xla_optimization_remark::Warning", tag="1")]
    pub warning: i32,
    /// Information such as which node was the problem.
    #[prost(string, tag="2")]
    pub debug_information: std::string::String,
}
pub mod xla_optimization_remark {
    /// Next ID: 6
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Warning {
        None = 0,
        InaccurateOperation = 1,
        SlowOperation = 2,
        UnimplementedOperation = 3,
        SlowImageResizeDimensions = 4,
        MegamorphicFunction = 5,
    }
}
