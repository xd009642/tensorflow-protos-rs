/// TensorId identifies a tensor in a TensorFlow graph, by specifying the output
/// index of a particular node in the graph.  If the output of the named node
/// feeds into other node(s), this corresponds to one or more edges.  Otherwise
/// it doesn't correspond to any existing edges at all, e.g. for output nodes.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorId {
    #[prost(string, tag="1")]
    pub node_name: std::string::String,
    #[prost(int64, tag="2")]
    pub output_index: i64,
}
/// Feed represents a single feed tensor in the graph, which corresponds to an
/// input argument for the generated computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Feed {
    #[prost(message, optional, tag="1")]
    pub id: ::std::option::Option<TensorId>,
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<super::TensorShapeProto>,
    /// Optional name for generated code.
    #[prost(string, tag="3")]
    pub name: std::string::String,
    /// Optional data type. This is not normally required, as the graph itself
    /// contains this information. However, if the node being fed is an op that is
    /// not linked into the binary, then the type cannot be inferred from the node;
    /// in this case, the type should be set here.
    #[prost(enumeration="super::DataType", tag="4")]
    pub r#type: i32,
}
/// Fetch represents a single fetch tensor in the graph, which corresponds to an
/// output argument for the generated computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Fetch {
    #[prost(message, optional, tag="1")]
    pub id: ::std::option::Option<TensorId>,
    /// Optional name for generated code.
    #[prost(string, tag="2")]
    pub name: std::string::String,
    /// Optional shape and data type. If specified, may be used for validation.
    #[prost(message, optional, tag="3")]
    pub shape: ::std::option::Option<super::TensorShapeProto>,
    #[prost(enumeration="super::DataType", tag="4")]
    pub r#type: i32,
}
/// Variable represents a resource variable with the given name, shape and type.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Variable {
    #[prost(string, tag="1")]
    pub node_name: std::string::String,
    /// Optional name for generated code. If empty, node_name will be used.
    #[prost(string, tag="2")]
    pub name: std::string::String,
    #[prost(message, optional, tag="3")]
    pub shape: ::std::option::Option<super::TensorShapeProto>,
    #[prost(enumeration="super::DataType", tag="4")]
    pub r#type: i32,
    /// Flag for variables that are never assigned. Assigments to a read-only
    /// variable or unassigned variables that are not read-only are invalid.
    #[prost(bool, tag="5")]
    pub readonly: bool,
}
/// Options used during the conversion and compilation process.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversionOptions {
    /// When true tf.fake_quant_* ops will be emitted as custom calls to a
    /// 'fake_quant_with_min_max_vars' function accepting the input, min, max,
    /// num_bits, and narrow_range values as runtime arguments.
    #[prost(bool, tag="1")]
    pub custom_fake_quant_op_calls: bool,
}
/// Config represents configuration information for tf2xla conversion.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Config {
    /// Each feed is a positional input argument for the generated computation.
    /// The order of each entry matches the order of each input argument.
    #[prost(message, repeated, tag="1")]
    pub feed: ::std::vec::Vec<Feed>,
    /// Each fetch is a positional output argument for the generated computation.
    /// The order of each entry matches the order of each output argument.
    #[prost(message, repeated, tag="2")]
    pub fetch: ::std::vec::Vec<Fetch>,
    /// Each variable is a named input and output of the generated computation.
    #[prost(message, repeated, tag="3")]
    pub variable: ::std::vec::Vec<Variable>,
    /// Optional conversion options.
    #[prost(message, optional, tag="4")]
    pub conversion_options: ::std::option::Option<ConversionOptions>,
}
/// TensorMetadata indicates the type and shape of a Tensor that is
/// part of a host compute transfer.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorMetadata {
    #[prost(enumeration="super::DataType", tag="1")]
    pub r#type: i32,
    #[prost(message, optional, tag="2")]
    pub shape: ::std::option::Option<super::TensorShapeProto>,
}
/// HostTransferMetadata describes a transfer either from host to device
/// or device to host. It has a key that is unique to the computation,
/// and metadata about the list of tensors being transferred.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HostTransferMetadata {
    /// The key used to identify this transfer.
    #[prost(string, tag="1")]
    pub key: std::string::String,
    /// For each Tensor being transferred, its type and shape.
    #[prost(message, repeated, tag="2")]
    pub metadata: ::std::vec::Vec<TensorMetadata>,
}
/// HostComputeMetadata describes all the sends and recvs
/// from all host compute transfer ops in a computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HostComputeMetadata {
    /// Metadata about each device_to_host transfer
    #[prost(message, repeated, tag="1")]
    pub device_to_host: ::std::vec::Vec<HostTransferMetadata>,
    /// Metadata about each host_to_device transfer
    #[prost(message, repeated, tag="2")]
    pub host_to_device: ::std::vec::Vec<HostTransferMetadata>,
}
