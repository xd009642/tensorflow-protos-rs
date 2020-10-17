#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoteTensorHandle {
    /// The ID of the operation that produced this tensor.
    #[prost(int64, tag="1")]
    pub op_id: i64,
    /// The index into the outputs of the operation that produced this tensor.
    #[prost(int32, tag="2")]
    pub output_num: i32,
    /// Device of the operation that produced this tensor. Cannot be empty.
    /// For multi-device functions, it's the default device passed to placer.
    #[prost(string, tag="3")]
    pub device: std::string::String,
    /// Device where the tensor is located. Can be empty if the operation producing
    /// this tensor is a multi-device function.
    #[prost(string, tag="4")]
    pub op_device: std::string::String,
    /// Tensor type.
    #[prost(enumeration="super::DataType", tag="5")]
    pub dtype: i32,
}
/// A proto representation of an eager operation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Operation {
    /// A unique identifier for the operation. Set by the client so that the client
    /// can uniquely identify the outputs of the scheduled operation.
    ///
    /// In the initial implementation, sending duplicate IDs has undefined
    /// behaviour, but additional constraints may be placed upon this in the
    /// future.
    #[prost(int64, tag="1")]
    pub id: i64,
    #[prost(string, tag="2")]
    pub name: std::string::String,
    #[prost(message, repeated, tag="3")]
    pub inputs: ::std::vec::Vec<RemoteTensorHandle>,
    /// Control Operation IDs that will be respected when ops are re-ordered by
    /// async execution. If async execution (+ op re-ordering) is not enabled, this
    /// should have no effect.
    #[prost(int64, repeated, tag="4")]
    pub control_op_ids: ::std::vec::Vec<i64>,
    #[prost(map="string, message", tag="5")]
    pub attrs: ::std::collections::HashMap<std::string::String, super::AttrValue>,
    #[prost(string, tag="6")]
    pub device: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct QueueItem {
    /// The remote executor should be able to handle either executing ops directly,
    /// or releasing any unused tensor handles, since the tensor lifetime is
    /// maintained by the client.
    #[prost(oneof="queue_item::Item", tags="1, 2, 3")]
    pub item: ::std::option::Option<queue_item::Item>,
}
pub mod queue_item {
    /// The remote executor should be able to handle either executing ops directly,
    /// or releasing any unused tensor handles, since the tensor lifetime is
    /// maintained by the client.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Item {
        #[prost(message, tag="1")]
        HandleToDecref(super::RemoteTensorHandle),
        #[prost(message, tag="2")]
        Operation(super::Operation),
        #[prost(message, tag="3")]
        SendTensor(super::SendTensorOp),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct QueueResponse {
    #[prost(message, repeated, tag="1")]
    pub shape: ::std::vec::Vec<super::TensorShapeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateContextRequest {
    /// Identifies the full cluster, and this particular worker's position within.
    #[prost(message, optional, tag="1")]
    pub server_def: ::std::option::Option<super::ServerDef>,
    /// Whether the ops on the worker should be executed synchronously or
    /// asynchronously. By default, ops are executed synchronously.
    #[prost(bool, tag="2")]
    pub r#async: bool,
    /// Number of seconds to keep the context alive. If more than keep_alive_secs
    /// has passed since a particular context has been communicated with, it will
    /// be garbage collected.
    #[prost(int64, tag="3")]
    pub keep_alive_secs: i64,
    /// This is the version for all the ops that will be enqueued by the client.
    #[prost(message, optional, tag="4")]
    pub version_def: ::std::option::Option<super::VersionDef>,
    /// Device attributes in the cluster
    #[prost(message, repeated, tag="6")]
    pub cluster_device_attributes: ::std::vec::Vec<super::DeviceAttributes>,
    /// The ID of the created context. This is usually a randomly generated number,
    /// that will be used to identify the context in future requests to the
    /// service. Contexts are not persisted through server restarts.
    /// This ID will be used for all future communications as well. It is essential
    /// that both ends use this ID for selecting a rendezvous to get everything to
    /// match.
    #[prost(fixed64, tag="7")]
    pub context_id: u64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateContextResponse {
    /// List of devices that are locally accessible to the worker.
    #[prost(message, repeated, tag="2")]
    pub device_attributes: ::std::vec::Vec<super::DeviceAttributes>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EnqueueRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
    #[prost(message, repeated, tag="3")]
    pub queue: ::std::vec::Vec<QueueItem>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EnqueueResponse {
    /// A single operation response for every item in the request.
    #[prost(message, repeated, tag="1")]
    pub queue_response: ::std::vec::Vec<QueueResponse>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WaitQueueDoneRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
    /// Ids to wait on. If empty, wait on everything currently pending.
    #[prost(int64, repeated, tag="2")]
    pub op_id: ::std::vec::Vec<i64>,
}
/// TODO(nareshmodi): Consider adding NodeExecStats here to be able to
/// propagate some stats.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WaitQueueDoneResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KeepAliveRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KeepAliveResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CloseContextRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CloseContextResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterFunctionRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
    #[prost(message, optional, tag="2")]
    pub function_def: ::std::option::Option<super::FunctionDef>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RegisterFunctionResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SendTensorOp {
    /// All remote tensors are identified by <Op ID, Output num>. To mimic this
    /// situation when directly sending tensors, we include an "artificial" op ID
    /// (which would have corresponded to the _Recv op when not using SendTensor).
    #[prost(int64, tag="1")]
    pub op_id: i64,
    /// The index within the repeated field is the output number that will help
    /// uniquely identify (along with the above op_id) the particular tensor.
    #[prost(message, repeated, tag="2")]
    pub tensors: ::std::vec::Vec<super::TensorProto>,
    /// The device on which the tensors should be resident.
    #[prost(string, tag="3")]
    pub device_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SendTensorRequest {
    #[prost(fixed64, tag="1")]
    pub context_id: u64,
    /// All remote tensors are identified by <Op ID, Output num>. To mimic this
    /// situation when directly sending tensors, we include an "artificial" op ID
    /// (which would have corresponded to the _Recv op when not using SendTensor).
    #[prost(int64, tag="2")]
    pub op_id: i64,
    /// The index within the repeated field is the output number that will help
    /// uniquely identify (along with the above op_id) the particular tensor.
    #[prost(message, repeated, tag="3")]
    pub tensors: ::std::vec::Vec<super::TensorProto>,
    /// The device on which the tensors should be resident.
    #[prost(string, tag="4")]
    pub device_name: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SendTensorResponse {
}
