/// It specifies the Python callstack that creates an op.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CodeDef {
    #[prost(message, repeated, tag="1")]
    pub traces: ::std::vec::Vec<code_def::Trace>,
}
pub mod code_def {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Trace {
        /// deprecated by file_id.
        #[prost(string, tag="1")]
        pub file: std::string::String,
        #[prost(int64, tag="6")]
        pub file_id: i64,
        #[prost(int32, tag="2")]
        pub lineno: i32,
        /// deprecated by function_id.
        #[prost(string, tag="3")]
        pub function: std::string::String,
        #[prost(int64, tag="7")]
        pub function_id: i64,
        /// deprecated line_id.
        #[prost(string, tag="4")]
        pub line: std::string::String,
        #[prost(int64, tag="8")]
        pub line_id: i64,
        #[prost(int32, tag="5")]
        pub func_start_line: i32,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpLogEntry {
    /// op name.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// float_ops is filled by tfprof Python API when called. It requires the
    /// op has RegisterStatistics defined. Currently, Conv2D, MatMul, etc, are
    /// implemented.
    #[prost(int64, tag="2")]
    pub float_ops: i64,
    /// User can define extra op type information for an op. This allows the user
    /// to select a group of ops precisely using op_type as a key.
    #[prost(string, repeated, tag="3")]
    pub types: ::std::vec::Vec<std::string::String>,
    /// Used to support tfprof "code" view.
    #[prost(message, optional, tag="4")]
    pub code_def: ::std::option::Option<CodeDef>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpLogProto {
    #[prost(message, repeated, tag="1")]
    pub log_entries: ::std::vec::Vec<OpLogEntry>,
    /// Maps from id of CodeDef file,function,line to its string
    /// In the future can also map other id of other fields to string.
    #[prost(map="int64, string", tag="2")]
    pub id_to_string: ::std::collections::HashMap<i64, std::string::String>,
}
/// A proto representation of the profiler's profile.
/// It allows serialization, shipping around and deserialization of the profiles.
///
/// Please don't depend on the internals of the profile proto.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileProto {
    #[prost(map="int64, message", tag="1")]
    pub nodes: ::std::collections::HashMap<i64, ProfileNode>,
    /// Whether or not has code traces.
    #[prost(bool, tag="2")]
    pub has_trace: bool,
    /// Whether or not the TF device tracer fails to return accelerator
    /// information (which could lead to 0 accelerator execution time).
    #[prost(bool, tag="5")]
    pub miss_accelerator_stream: bool,
    /// Traced steps.
    #[prost(int64, repeated, tag="3")]
    pub steps: ::std::vec::Vec<i64>,
    /// Maps from id of CodeDef file,function,line to its string
    /// In the future can also map other id of other fields to string.
    #[prost(map="int64, string", tag="4")]
    pub id_to_string: ::std::collections::HashMap<i64, std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProfileNode {
    /// graph node name.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// graph operation type.
    #[prost(string, tag="9")]
    pub op: std::string::String,
    /// A unique id for the node.
    #[prost(int64, tag="13")]
    pub id: i64,
    #[prost(map="int32, int64", tag="2")]
    pub inputs: ::std::collections::HashMap<i32, i64>,
    #[prost(map="int32, message", tag="16")]
    pub input_shapes: ::std::collections::HashMap<i32, Tuple>,
    #[prost(map="int32, int64", tag="3")]
    pub outputs: ::std::collections::HashMap<i32, i64>,
    #[prost(map="int32, message", tag="15")]
    pub output_shapes: ::std::collections::HashMap<i32, Tuple>,
    /// A map from source node id to its output index to current node.
    #[prost(map="int64, int32", tag="14")]
    pub src_output_index: ::std::collections::HashMap<i64, i32>,
    #[prost(int64, repeated, tag="4")]
    pub shape: ::std::vec::Vec<i64>,
    #[prost(string, repeated, tag="5")]
    pub op_types: ::std::vec::Vec<std::string::String>,
    #[prost(string, tag="6")]
    pub canonical_device: std::string::String,
    #[prost(string, tag="7")]
    pub host_device: std::string::String,
    #[prost(int64, tag="8")]
    pub float_ops: i64,
    #[prost(message, optional, tag="10")]
    pub trace: ::std::option::Option<CodeDef>,
    #[prost(map="string, message", tag="11")]
    pub attrs: ::std::collections::HashMap<std::string::String, super::AttrValue>,
    #[prost(map="int64, message", tag="12")]
    pub execs: ::std::collections::HashMap<i64, ExecProfile>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecProfile {
    /// Can be larger than 1 if run multiple times in loop.
    #[prost(int64, tag="1")]
    pub run_count: i64,
    /// The earliest/latest time including scheduling and execution.
    #[prost(int64, tag="2")]
    pub all_start_micros: i64,
    #[prost(int64, tag="3")]
    pub latest_end_micros: i64,
    /// device -> vector of {op_start_micros, op_exec_micros} pairs.
    /// accelerator_execs: gpu:id/stream:all -> {op_start_micros, op_exec_micros}
    /// For accelerator, vector size can be larger than 1, multiple kernel fires
    /// or in tf.while_loop.
    #[prost(map="string, message", tag="4")]
    pub accelerator_execs: ::std::collections::HashMap<std::string::String, ExecTime>,
    /// cpu_execs: cpu/gpu:id -> {op_start_micros, op_exec_micros}
    /// For cpu, vector size can be larger than 1 if in tf.while_loop.
    #[prost(map="string, message", tag="5")]
    pub cpu_execs: ::std::collections::HashMap<std::string::String, ExecTime>,
    /// Each entry to memory information of a scheduling of the node.
    /// Normally, there will be multiple entries in while_loop.
    #[prost(message, repeated, tag="7")]
    pub memory_execs: ::std::vec::Vec<ExecMemory>,
    /// The allocation and deallocation times and sizes throughout execution.
    #[prost(message, repeated, tag="11")]
    pub allocations: ::std::vec::Vec<super::AllocationRecord>,
    /// The devices related to this execution.
    #[prost(string, repeated, tag="6")]
    pub devices: ::std::vec::Vec<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecTime {
    #[prost(message, repeated, tag="1")]
    pub times: ::std::vec::Vec<Tuple>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecMemory {
    /// This is the timestamp when the memory information was tracked.
    #[prost(int64, tag="1")]
    pub memory_micros: i64,
    /// NOTE: Please don't depend on the following 4 fields yet. Due to
    /// TensorFlow internal tracing issues, the numbers can be quite wrong.
    /// TODO(xpan): Fix the TensorFlow internal tracing.
    #[prost(int64, tag="2")]
    pub host_temp_bytes: i64,
    #[prost(int64, tag="3")]
    pub host_persistent_bytes: i64,
    #[prost(int64, tag="4")]
    pub accelerator_temp_bytes: i64,
    #[prost(int64, tag="5")]
    pub accelerator_persistent_bytes: i64,
    /// Total bytes requested by the op.
    #[prost(int64, tag="6")]
    pub requested_bytes: i64,
    /// Total bytes requested by the op and released before op end.
    #[prost(int64, tag="7")]
    pub peak_bytes: i64,
    /// Total bytes requested by the op and not released after op end.
    #[prost(int64, tag="8")]
    pub residual_bytes: i64,
    /// Total bytes output by the op (not necessarily requested by the op).
    #[prost(int64, tag="9")]
    pub output_bytes: i64,
    /// The total number of bytes currently allocated by the allocator if >0.
    #[prost(int64, tag="10")]
    pub allocator_bytes_in_use: i64,
    /// The memory of each output of the operation.
    #[prost(map="int32, message", tag="11")]
    pub output_memory: ::std::collections::HashMap<i32, Memory>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Tuple {
    #[prost(int64, repeated, tag="1")]
    pub int64_values: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Memory {
    #[prost(int64, tag="1")]
    pub bytes: i64,
    #[prost(uint64, tag="2")]
    pub ptr: u64,
}
/// Refers to tfprof_options.h/cc for documentation.
/// Only used to pass tfprof options from Python to C++.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OptionsProto {
    #[prost(int64, tag="1")]
    pub max_depth: i64,
    #[prost(int64, tag="2")]
    pub min_bytes: i64,
    #[prost(int64, tag="19")]
    pub min_peak_bytes: i64,
    #[prost(int64, tag="20")]
    pub min_residual_bytes: i64,
    #[prost(int64, tag="21")]
    pub min_output_bytes: i64,
    #[prost(int64, tag="3")]
    pub min_micros: i64,
    #[prost(int64, tag="22")]
    pub min_accelerator_micros: i64,
    #[prost(int64, tag="23")]
    pub min_cpu_micros: i64,
    #[prost(int64, tag="4")]
    pub min_params: i64,
    #[prost(int64, tag="5")]
    pub min_float_ops: i64,
    #[prost(int64, tag="17")]
    pub min_occurrence: i64,
    #[prost(int64, tag="18")]
    pub step: i64,
    #[prost(string, tag="7")]
    pub order_by: std::string::String,
    #[prost(string, repeated, tag="8")]
    pub account_type_regexes: ::std::vec::Vec<std::string::String>,
    #[prost(string, repeated, tag="9")]
    pub start_name_regexes: ::std::vec::Vec<std::string::String>,
    #[prost(string, repeated, tag="10")]
    pub trim_name_regexes: ::std::vec::Vec<std::string::String>,
    #[prost(string, repeated, tag="11")]
    pub show_name_regexes: ::std::vec::Vec<std::string::String>,
    #[prost(string, repeated, tag="12")]
    pub hide_name_regexes: ::std::vec::Vec<std::string::String>,
    #[prost(bool, tag="13")]
    pub account_displayed_op_only: bool,
    #[prost(string, repeated, tag="14")]
    pub select: ::std::vec::Vec<std::string::String>,
    #[prost(string, tag="15")]
    pub output: std::string::String,
    #[prost(string, tag="16")]
    pub dump_to_file: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AdvisorOptionsProto {
    /// checker name -> a dict of key-value options.
    #[prost(map="string, message", tag="1")]
    pub checkers: ::std::collections::HashMap<std::string::String, advisor_options_proto::CheckerOption>,
}
pub mod advisor_options_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct CheckerOption {
        #[prost(map="string, string", tag="1")]
        pub options: ::std::collections::HashMap<std::string::String, std::string::String>,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfProfTensorProto {
    #[prost(enumeration="super::DataType", tag="1")]
    pub dtype: i32,
    /// Flatten tensor in row-major.
    /// Only one of the following array is set.
    #[prost(double, repeated, tag="2")]
    pub value_double: ::std::vec::Vec<f64>,
    #[prost(int64, repeated, tag="3")]
    pub value_int64: ::std::vec::Vec<i64>,
    #[prost(string, repeated, tag="4")]
    pub value_str: ::std::vec::Vec<std::string::String>,
}
/// A node in TensorFlow graph. Used by scope/graph view.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphNodeProto {
    /// op name.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// tensor value restored from checkpoint.
    #[prost(message, optional, tag="15")]
    pub tensor_value: ::std::option::Option<TfProfTensorProto>,
    /// op execution time.
    /// A node can be defined once but run multiple times in tf.while_loop.
    /// the times sum up all different runs.
    #[prost(int64, tag="21")]
    pub run_count: i64,
    #[prost(int64, tag="2")]
    pub exec_micros: i64,
    #[prost(int64, tag="17")]
    pub accelerator_exec_micros: i64,
    #[prost(int64, tag="18")]
    pub cpu_exec_micros: i64,
    /// Total bytes requested by the op.
    #[prost(int64, tag="3")]
    pub requested_bytes: i64,
    /// Max bytes allocated and being used by the op at a point.
    #[prost(int64, tag="24")]
    pub peak_bytes: i64,
    /// Total bytes requested by the op and not released before end.
    #[prost(int64, tag="25")]
    pub residual_bytes: i64,
    /// Total bytes output by the op (not necessarily allocated by the op).
    #[prost(int64, tag="26")]
    pub output_bytes: i64,
    /// Number of parameters if available.
    #[prost(int64, tag="4")]
    pub parameters: i64,
    /// Number of float operations.
    #[prost(int64, tag="13")]
    pub float_ops: i64,
    /// Device the op is assigned to.
    /// Since an op can fire multiple kernel calls, there can be multiple devices.
    #[prost(string, repeated, tag="10")]
    pub devices: ::std::vec::Vec<std::string::String>,
    /// The following are the aggregated stats from all *accounted* children and
    /// the node itself. The actual children depend on the data structure used.
    /// In graph view, children are inputs recursively.
    /// In scope view, children are nodes under the name scope.
    #[prost(int64, tag="23")]
    pub total_definition_count: i64,
    #[prost(int64, tag="22")]
    pub total_run_count: i64,
    #[prost(int64, tag="6")]
    pub total_exec_micros: i64,
    #[prost(int64, tag="19")]
    pub total_accelerator_exec_micros: i64,
    #[prost(int64, tag="20")]
    pub total_cpu_exec_micros: i64,
    #[prost(int64, tag="7")]
    pub total_requested_bytes: i64,
    #[prost(int64, tag="27")]
    pub total_peak_bytes: i64,
    #[prost(int64, tag="28")]
    pub total_residual_bytes: i64,
    #[prost(int64, tag="29")]
    pub total_output_bytes: i64,
    #[prost(int64, tag="8")]
    pub total_parameters: i64,
    #[prost(int64, tag="14")]
    pub total_float_ops: i64,
    /// shape information, if available.
    /// TODO(xpan): Why is this repeated?
    #[prost(message, repeated, tag="11")]
    pub shapes: ::std::vec::Vec<super::TensorShapeProto>,
    #[prost(map="int32, message", tag="16")]
    pub input_shapes: ::std::collections::HashMap<i32, super::TensorShapeProto>,
    /// Descendants of the graph. The actual descendants depend on the data
    /// structure used (scope, graph).
    #[prost(message, repeated, tag="12")]
    pub children: ::std::vec::Vec<GraphNodeProto>,
}
/// A node that groups multiple GraphNodeProto.
/// Depending on the 'view', the semantics of the TFmultiGraphNodeProto
/// is different:
/// code view: A node groups all TensorFlow graph nodes created by the
///            Python code.
/// op view:   A node groups all TensorFlow graph nodes that are of type
///            of the op (e.g. MatMul, Conv2D).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MultiGraphNodeProto {
    /// Name of the node.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// code execution time.
    #[prost(int64, tag="2")]
    pub exec_micros: i64,
    #[prost(int64, tag="12")]
    pub accelerator_exec_micros: i64,
    #[prost(int64, tag="13")]
    pub cpu_exec_micros: i64,
    /// Total requested bytes by the code.
    #[prost(int64, tag="3")]
    pub requested_bytes: i64,
    /// Max bytes allocated and being used by the op at a point.
    #[prost(int64, tag="16")]
    pub peak_bytes: i64,
    /// Total bytes requested by the op and not released before end.
    #[prost(int64, tag="17")]
    pub residual_bytes: i64,
    /// Total bytes output by the op (not necessarily allocated by the op).
    #[prost(int64, tag="18")]
    pub output_bytes: i64,
    /// Number of parameters if available.
    #[prost(int64, tag="4")]
    pub parameters: i64,
    /// Number of float operations.
    #[prost(int64, tag="5")]
    pub float_ops: i64,
    /// The following are the aggregated stats from descendants.
    /// The actual descendants depend on the data structure used.
    #[prost(int64, tag="6")]
    pub total_exec_micros: i64,
    #[prost(int64, tag="14")]
    pub total_accelerator_exec_micros: i64,
    #[prost(int64, tag="15")]
    pub total_cpu_exec_micros: i64,
    #[prost(int64, tag="7")]
    pub total_requested_bytes: i64,
    #[prost(int64, tag="19")]
    pub total_peak_bytes: i64,
    #[prost(int64, tag="20")]
    pub total_residual_bytes: i64,
    #[prost(int64, tag="21")]
    pub total_output_bytes: i64,
    #[prost(int64, tag="8")]
    pub total_parameters: i64,
    #[prost(int64, tag="9")]
    pub total_float_ops: i64,
    /// TensorFlow graph nodes contained by the MultiGraphNodeProto.
    #[prost(message, repeated, tag="10")]
    pub graph_nodes: ::std::vec::Vec<GraphNodeProto>,
    /// Descendants of the node. The actual descendants depend on the data
    /// structure used.
    #[prost(message, repeated, tag="11")]
    pub children: ::std::vec::Vec<MultiGraphNodeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AdviceProto {
    /// checker name -> a list of reports from the checker.
    #[prost(map="string, message", tag="1")]
    pub checkers: ::std::collections::HashMap<std::string::String, advice_proto::Checker>,
}
pub mod advice_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Checker {
        #[prost(string, repeated, tag="2")]
        pub reports: ::std::vec::Vec<std::string::String>,
    }
}
