#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceAssignment {
    /// As many ComputationDevice as many there are computations (number
    /// of cores per replica).
    #[prost(message, repeated, tag="1")]
    pub computation_devices: ::std::vec::Vec<device_assignment::ComputationDevice>,
}
pub mod device_assignment {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ComputationDevice {
        /// As many replicas as there are in the replicated computation.
        #[prost(message, repeated, tag="1")]
        pub replica_devices: ::std::vec::Vec<computation_device::DeviceMeshCoordinates>,
    }
    pub mod computation_device {
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct DeviceMeshCoordinates {
            /// The mesh coordinates for the device. Usually (X, Y, Core), in the order
            /// in which they are returned in the TopologyProto.
            ///  X    = value(0)
            ///  Y    = value(1)
            ///  Core = value(2)
            #[prost(int32, repeated, tag="1")]
            pub value: ::std::vec::Vec<i32>,
        }
    }
}
/// Options for an XLA compilation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaComputationConfig {
    /// The number of replicas the computation will be run on. If this is
    /// default (0) it is interpreted as 1.
    #[prost(int32, tag="1")]
    pub num_replicas: i32,
    /// The number of "model-parallel" cores per replica. If this is
    /// default (0) it is interpreted as 1.
    #[prost(int32, tag="2")]
    pub num_cores_per_replica: i32,
    /// Optional metadata about host sends and recvs.
    #[prost(message, optional, tag="3")]
    pub host_compute_metadata: ::std::option::Option<super::tensorflow::tf2xla::HostComputeMetadata>,
    /// The arg/result shapes for the whole computation.
    #[prost(message, optional, tag="4")]
    pub program_shape: ::std::option::Option<super::xla::ProgramShapeProto>,
    /// The arg/result shapes for each core of a model-parallel
    /// computation. per_core_args_and_result_shapes is optional for a
    /// single-core computation.
    #[prost(message, repeated, tag="5")]
    pub per_core_program_shape: ::std::vec::Vec<super::xla::ProgramShapeProto>,
    /// Describes how replicated computation instances should be assigned to
    /// devices. There are num_cores_per_replica computations, and each one will be
    /// sent and executed to the set of replica device numbers described in the
    /// DeviceAssignment proto.
    #[prost(message, optional, tag="6")]
    pub device_assignment: ::std::option::Option<DeviceAssignment>,
    /// The debugging options to be passed to the XLA compilation process.
    #[prost(message, optional, tag="7")]
    pub debug_options: ::std::option::Option<super::xla::DebugOptions>,
}
/// Options and XLA computation for a compilation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaComputation {
    #[prost(message, optional, tag="1")]
    pub config: ::std::option::Option<XlaComputationConfig>,
    #[prost(message, optional, tag="2")]
    pub hlo_snapshot: ::std::option::Option<super::xla::HloSnapshot>,
}
/// Literal to allocate space for, and transfer to, device memory.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaAllocation {
    #[prost(message, optional, tag="2")]
    pub value: ::std::option::Option<super::xla::LiteralProto>,
}
/// Node in a tree describing a tuple constructed from input handles. A
/// node is an internal node if tuples is non-empty, in which case
/// input_index and release_input_handle are ignored. Otherwise a node
/// is a leaf node. Each leaf XLATupleNode is the index of an input
/// which corresponds to a handle that will be grafted onto the output
/// tuple at that location. If release_input_handle is true that input
/// handle will be released and become invalid.  Inputs may be repeated
/// in which case leaves of the output tuple will alias. If an input is
/// repeated, release_input_handle must be false for every leaf where
/// that input appears.
///
/// For example, if input 0 has shape {} and input 1 has shape {2,3}
/// then the XLATupleNode with structure {1,{0,1}} corresponds to a
/// tuple with shape {{2,3},{{},{2,3}}}.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XlaTupleNode {
    #[prost(int32, tag="1")]
    pub input_index: i32,
    #[prost(bool, tag="2")]
    pub release_input_handle: bool,
    #[prost(message, repeated, tag="3")]
    pub tuples: ::std::vec::Vec<XlaTupleNode>,
}
/// Options for an XLA execution.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XrtExecutionConfig {
    /// Local device to run on. This is present because the execute Op
    /// may be placed on a device such as CPU or TPU_SYSTEM that
    /// logically manages multiple cores.
    #[prost(int32, tag="1")]
    pub device_ordinal: i32,
    /// Which model-parallel computation to run from the compiled bundle.
    #[prost(int32, tag="2")]
    pub core_index_in_replica: i32,
    /// Optional key to disambiguate between executions. This is only
    /// needed if multiple host send/recvs may be outstanding
    /// concurrently with executions.
    #[prost(string, tag="3")]
    pub execution_instance_key: std::string::String,
    /// If non-zero, rng_seed to reset the core with.
    #[prost(uint32, tag="4")]
    pub rng_seed: u32,
    /// If true, release allocation handles on the inputs after running.
    #[prost(bool, tag="5")]
    pub release_input_handles: bool,
    /// If true, release the handle to the computation after running.
    #[prost(bool, tag="6")]
    pub release_compilation_handle: bool,
    /// If set to true, and the result shape is a tuple, then instead of returning
    /// a single tuple allocation the execution will return a vector of
    /// allocations, one for each of the first-level elements of the result tuple.
    #[prost(bool, tag="7")]
    pub return_exploded_tuple: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XrtChainedExecuteConfig {
    /// If non-zero, rng_seed to reset the core with.
    #[prost(uint32, tag="1")]
    pub rng_seed: u32,
    /// Which model-parallel computation to run from the compiled bundle.
    #[prost(int32, tag="2")]
    pub core_index_in_replica: i32,
    /// Optional key to disambiguate between executions. This is only needed if
    /// multiple host send/recvs may be outstanding concurrently with executions.
    #[prost(string, tag="3")]
    pub execution_instance_key: std::string::String,
}
/// A single chained execute operation. An operation can either be a device data
/// load, or an existing (as in, previously compiled and accessible via its int64
/// handle) XLA computation execution.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XrtChainedExecuteOp {
    /// The outputs of this XRTChainedExecuteOp operation.
    #[prost(message, repeated, tag="3")]
    pub outputs: ::std::vec::Vec<xrt_chained_execute_op::Output>,
    /// The inputs of this XRTChainedExecuteOp operation. If data_handle is set,
    /// there are no inputs.
    #[prost(message, repeated, tag="4")]
    pub inputs: ::std::vec::Vec<xrt_chained_execute_op::Input>,
    #[prost(oneof="xrt_chained_execute_op::OpOneof", tags="1, 2")]
    pub op_oneof: ::std::option::Option<xrt_chained_execute_op::OpOneof>,
}
pub mod xrt_chained_execute_op {
    /// Represents an input for this operation.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Input {
        /// The index within the XRTChainedExecutePlan.ops post-order of the source
        /// operation for this input.
        #[prost(int64, tag="1")]
        pub op_index: i64,
        /// The output index of the value generated by the operation at op_index.
        /// Zero (default value) means no index ({}) while if an indexing is
        /// required, output_index needs to be set to index+1.
        /// Thanks proto3!
        #[prost(int64, tag="2")]
        pub output_index: i64,
    }
    /// Represents an output of the XRTChainedExecute operation, which should
    /// originate by the output of this operation.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Output {
        /// The index in the value generated by this operation, which should be
        /// forwarded as XRTChainedExecute output. If output_index is zero (default
        /// value) the whole output will be used as result. This means that if the
        /// output shape is a tuple, the result will be the full tuple. Otherwise the
        /// real sub-tuple index will be output_index - 1.
        #[prost(int64, tag="1")]
        pub output_index: i64,
        /// The index in the vector of the results returned by the XRTChainedExecute
        /// operation, where this output should be forwarded.
        #[prost(int64, tag="2")]
        pub result_index: i64,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum OpOneof {
        /// The handle to an existing XRT device data.
        #[prost(int64, tag="1")]
        DataHandle(i64),
        /// The handle to an existing XRT compiled computation.
        #[prost(int64, tag="2")]
        ComputationHandle(i64),
    }
}
/// Execution plan for the XRTChainedExecute operation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct XrtChainedExecutePlan {
    /// The post order with the XRT computations to be executed.
    #[prost(message, repeated, tag="1")]
    pub ops: ::std::vec::Vec<XrtChainedExecuteOp>,
}
