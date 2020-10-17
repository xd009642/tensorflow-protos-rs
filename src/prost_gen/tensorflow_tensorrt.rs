/// Containing information for a serialized TensorRT engine.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrtEngineInstance {
    /// The input shapes of the TRT engine.
    #[prost(message, repeated, tag="1")]
    pub input_shapes: ::std::vec::Vec<super::TensorShapeProto>,
    /// The serialized TRT engine.
    ///
    /// TODO(laigd): consider using a more efficient in-memory representation
    /// instead of string which is the default here.
    #[prost(bytes, tag="2")]
    pub serialized_engine: std::vec::Vec<u8>,
}
