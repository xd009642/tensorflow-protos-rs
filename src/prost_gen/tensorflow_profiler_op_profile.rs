/// Profile is the top-level data that summarizes a program.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Profile {
    /// Root of a profile broken down by instruction category.
    #[prost(message, optional, tag="1")]
    pub by_category: ::std::option::Option<Node>,
    /// Root of a profile broken down by program.
    #[prost(message, optional, tag="4")]
    pub by_program: ::std::option::Option<Node>,
}
/// An entry in the profile tree. (An instruction, or set of instructions).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Node {
    /// Semantics depend on contents.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// May be omitted e.g. for fused instructions.
    #[prost(message, optional, tag="2")]
    pub metrics: ::std::option::Option<Metrics>,
    /// Subjected to pruning.
    #[prost(message, repeated, tag="3")]
    pub children: ::std::vec::Vec<Node>,
    /// Total number of children before pruning.
    #[prost(int32, tag="6")]
    pub num_children: i32,
    /// Details about what this node represents.
    #[prost(oneof="node::Contents", tags="4, 5")]
    pub contents: ::std::option::Option<node::Contents>,
}
pub mod node {
    /// A category of XLA instructions.
    /// name is a descriptive string, like "data formatting".
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InstructionCategory {
    }
    /// A single XLA instruction.
    /// name is the unique instruction id, like "%multiply.5".
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct XlaInstruction {
        /// Opcode like %multiply
        #[prost(string, tag="1")]
        pub op: std::string::String,
        /// %multiply = [shape]multiply(operand1, operand2)
        #[prost(string, tag="2")]
        pub expression: std::string::String,
        /// Typically the TensorFlow operation name.
        #[prost(string, tag="3")]
        pub provenance: std::string::String,
        #[prost(string, tag="4")]
        pub category: std::string::String,
        /// Describes the physical memory layout of the instruction's primary input.
        /// e.g. for a convolution, this analyzes the image and ignores the kernel.
        #[prost(message, optional, tag="5")]
        pub layout: ::std::option::Option<xla_instruction::LayoutAnalysis>,
    }
    pub mod xla_instruction {
        #[derive(Clone, PartialEq, ::prost::Message)]
        pub struct LayoutAnalysis {
            /// The physical data layout, from most-minor to most-major dimensions.
            #[prost(message, repeated, tag="1")]
            pub dimensions: ::std::vec::Vec<layout_analysis::Dimension>,
        }
        pub mod layout_analysis {
            #[derive(Clone, PartialEq, ::prost::Message)]
            pub struct Dimension {
                /// Size of the data in this dimension.
                #[prost(int32, tag="1")]
                pub size: i32,
                /// Data must be padded to a multiple of alignment.
                #[prost(int32, tag="2")]
                pub alignment: i32,
                /// What the dimension represents, e.g. "spatial".
                #[prost(string, tag="3")]
                pub semantics: std::string::String,
            }
        }
    }
    /// Details about what this node represents.
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Contents {
        #[prost(message, tag="4")]
        Category(InstructionCategory),
        #[prost(message, tag="5")]
        Xla(XlaInstruction),
    }
}
/// Measurements of an operation (or aggregated set of operations).
/// Metrics are always "total" rather than "self".
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Metrics {
    /// Core-time taken by this operation, as a fraction of all operations.
    #[prost(double, tag="1")]
    pub time: f64,
    /// Floating point computations performed by this operation, as a fraction of
    /// peak core FLOPS * program time. This representation has useful properties:
    ///  - it is proportional to the number of floating point operations performed
    ///  - utilization is flops/time
    ///  - wasted potential flops is proportional to time - flops
    ///  - it does not reveal the peak core FLOPS of the hardware
    #[prost(double, tag="2")]
    pub flops: f64,
    /// The memory bandwidth used to load operands, as a fraction of
    /// thereotical memory bandwidth on the specific hardware.
    #[prost(double, tag="3")]
    pub memory_bandwidth: f64,
    /// Elapsed core-time in picoseconds.
    #[prost(double, tag="11")]
    pub raw_time: f64,
    /// Total floating-point operations performed.
    #[prost(double, tag="12")]
    pub raw_flops: f64,
    /// Total bytes accessed (include read/write).
    #[prost(double, tag="13")]
    pub raw_bytes_accessed: f64,
}
