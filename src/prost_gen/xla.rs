/// Describes the padding configuration for Pad operation. The padding amount on
/// both edges as well as between the elements are specified for each dimension.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PaddingConfig {
    /// The padding configuration for all dimensions.
    #[prost(message, repeated, tag="1")]
    pub dimensions: ::std::vec::Vec<padding_config::PaddingConfigDimension>,
}
pub mod padding_config {
    /// Describes the padding configuration for a dimension.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct PaddingConfigDimension {
        /// Padding amount on the low-end (next to the index 0). May be negative.
        #[prost(int64, tag="1")]
        pub edge_padding_low: i64,
        /// Padding amount on the high-end (next to the highest index). May be
        /// negative.
        #[prost(int64, tag="2")]
        pub edge_padding_high: i64,
        /// Padding amount between the elements. May not be negative.
        #[prost(int64, tag="3")]
        pub interior_padding: i64,
    }
}
/// Describes a tile used in tiling-based layout. Refer to
/// g3doc/third_party/tensorflow/compiler/xla/g3doc/layout_with_tiling.md for
/// details about tiling-based layout.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TileProto {
    /// Number of elements in each dimension of the tile. It's ordered from the
    /// most major dimension of the tile to the most minor dimension of the tile.
    /// The dimensions correspond to a suffix of the dimensions of the shape being
    /// tiled.
    #[prost(int64, repeated, tag="1")]
    pub dimensions: ::std::vec::Vec<i64>,
}
/// A layout describes how the array is placed in (1D) memory space.  This
/// includes the minor-to-major ordering of dimensions within a shape.
///
/// Clients must specify the layouts of input Literals to the
/// computation. Layouts specified in interior operations which take Shapes (for
/// example, Convert) are ignored.
///
/// See the XLA documentation for more information on shapes and layouts.
///
/// LINT.IfChange
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LayoutProto {
    /// The method used to store the data in memory. The format determines which of
    /// the other fields are used by the layout.
    #[prost(enumeration="Format", tag="4")]
    pub format: i32,
    /// Sequence of dimension numbers, from minor (fastest varying index) to major
    /// (slowest varying index). This field is required.
    #[prost(int64, repeated, tag="1")]
    pub minor_to_major: ::std::vec::Vec<i64>,
    /// The maximum number of elements that can be stored for SPARSE formats.  This
    /// can be used to determine the maximum size in bytes of arrays stored in
    /// memory.  This field must be unset unless the format is SPARSE.
    #[prost(int64, tag="5")]
    pub max_sparse_elements: i64,
    /// A sequence of tiles, starting from the tile that's applied first to the
    /// Shape.
    ///
    /// TODO(b/119839262): implement tiling in each backend or add Unimplemented
    /// error.
    #[prost(message, repeated, tag="6")]
    pub tiles: ::std::vec::Vec<TileProto>,
    /// Bit size of each element. If the size is bigger than what the element
    /// type requires, the value is stored in the least significant
    /// bits and the additional most significant bits are filled with 0's.
    ///
    /// TODO(b/119839262): implement in each backend or add Unimplemented error.
    #[prost(int64, tag="7")]
    pub element_size_in_bits: i64,
    /// Memory space where this array resides. The integer field is interpreted in
    /// a backend-specific manner.
    #[prost(int64, tag="8")]
    pub memory_space: i64,
}
/// A shape describes the number of dimensions in the array, the size of each
/// dimension, and the primitive component type.
///
/// Tuples are a special case in that they have rank zero and have tuple_shapes
/// defined.
///
/// See the XLA documentation for more information on shapes and layouts.
///
/// LINT.IfChange
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShapeProto {
    /// The element type for this shape.
    #[prost(enumeration="PrimitiveType", tag="2")]
    pub element_type: i32,
    /// The size (number of elements) for each dimension, or an upper bound on the
    /// size if the dimension is dynamic.  In XLA, dimensions are numbered from 0
    /// to N-1 for an N-dimensional array. The first element of 'dimensions' is the
    /// size of dimension 0, the second element is the size of dimension 1, and so
    /// forth.  Empty list indicates a scalar.
    ///
    /// If the respective element in 'is_dimension_dynamic' is true then the value
    /// in this field represents an upper bound on the size of the dimension.
    #[prost(int64, repeated, tag="3")]
    pub dimensions: ::std::vec::Vec<i64>,
    /// For tuples only, the shapes of constituent shapes in the tuple sequence.
    #[prost(message, repeated, tag="4")]
    pub tuple_shapes: ::std::vec::Vec<ShapeProto>,
    /// The layout used to back this shape.
    #[prost(message, optional, tag="5")]
    pub layout: ::std::option::Option<LayoutProto>,
    /// For arrays, this indicates whether or not each dimension is
    /// dynamically-sized. The number of elements in this repeated field should be
    /// zero (indicating that no dimensions are dynamic) or equal to the number of
    /// elements in the 'dimensions' field.
    #[prost(bool, repeated, tag="6")]
    pub is_dynamic_dimension: ::std::vec::Vec<bool>,
}
/// Shape of the parameters and output of a computation (like a traditional
/// function signature).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProgramShapeProto {
    #[prost(message, repeated, tag="1")]
    pub parameters: ::std::vec::Vec<ShapeProto>,
    #[prost(message, optional, tag="2")]
    pub result: ::std::option::Option<ShapeProto>,
    #[prost(string, repeated, tag="3")]
    pub parameter_names: ::std::vec::Vec<std::string::String>,
}
/// Statistics of a computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputationStats {
    /// The number of floating point operations in the computation.
    #[prost(double, tag="1")]
    pub flop_count: f64,
    /// The number of transcendental operations (e.g., exp) in the computation.
    #[prost(double, tag="2")]
    pub transcendental_count: f64,
}
/// Symbolization metadata for HLO Instructions.
///
/// This metadata is used for debugging XLA code generation, as well as
/// performance profiling of XLA-generated executables.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpMetadata {
    /// The framework op name that generated this XLA op.
    ///
    /// Frameworks that build on top of XLA should mirror the names of their ops
    /// back to users by specifying the op_type. In this way, even if the
    /// framework's "ops" are implemented as multiple XLA HLO Ops, they can be
    /// grouped appropriately. (e.g. if a SoftMax layer is emitted into XLA as
    /// multiple ops, then each op should have the op_type be "SoftMax".)
    #[prost(string, tag="1")]
    pub op_type: std::string::String,
    /// The user-specified name of the op.
    ///
    /// This name is often unique within a computation. Note: some frameworks
    /// add auto-generated names if the user does not provide one.
    #[prost(string, tag="2")]
    pub op_name: std::string::String,
    /// Indicate a file and line that this op is associated to in a user's program.
    ///
    /// e.g. it could be the file and line of user code that generated the op.
    #[prost(string, tag="3")]
    pub source_file: std::string::String,
    #[prost(int32, tag="4")]
    pub source_line: i32,
}
/// Profile data from the execution of a computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionProfile {
    /// Whether the executable was read from the compilation cache.
    #[prost(bool, tag="1")]
    pub compilation_cache_hit: bool,
    /// The time in milliseconds spent to compile the computation. This only set if
    /// the executable was not read from the compilation cache
    /// (compilation_cache_hit == false).
    #[prost(int64, tag="2")]
    pub compile_time_ms: i64,
    /// The number of cycles spent for the computation. This does not include the
    /// time taken for the data transfers between the host and the device. This is
    /// a target-dependent field and only used for debugging purposes.
    #[prost(int64, tag="3")]
    pub compute_cycle_count: i64,
    /// The time in nanoseconds spent for the computation, without data transfer.
    #[prost(int64, tag="4")]
    pub compute_time_ns: i64,
    /// The time in nanoseconds spent for the entire computation, including the
    /// result data transfer time. Current implementation does not spend any cycles
    /// for the input data transfer since the memory is initialized with the proper
    /// values before the execution.
    #[prost(int64, tag="5")]
    pub compute_and_transfer_time_ns: i64,
    /// The size of the binary code in the executable.
    #[prost(int64, tag="6")]
    pub executable_size_in_bytes: i64,
    /// Whether this profile was drawn from a cache of profiles instead of from
    /// execution on the hardware.
    #[prost(bool, tag="7")]
    pub profile_cache_hit: bool,
}
/// Handle given to a user that represents an execution that the user launched
/// asynchronously on the device.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionHandle {
    #[prost(int64, tag="1")]
    pub handle: i64,
}
/// Handle given to a user that represents a globally accessible allocation.
/// Contrast this against a ComputationDataHandle, which is not globally
/// accessible, since it only exists within a specific computation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GlobalDataHandle {
    #[prost(int64, tag="1")]
    pub handle: i64,
}
/// Handle given to a user that represents a replicated virtual device. Each
/// replicated device represents N physical devices for execution where N is the
/// number of replicas.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceHandle {
    #[prost(int64, tag="1")]
    pub handle: i64,
    /// The number of model-parallel virtual devices that communicate via XLA
    /// Send/Recv instructions.
    #[prost(int64, tag="2")]
    pub device_count: i64,
}
/// Handle given to a user to represent a channel between two computations
/// via a Send and Recv instruction pair. Channels are unbuffered, so Send
/// Send instructions will be blocked until the data is transferred.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ChannelHandle {
    #[prost(int64, tag="1")]
    pub handle: i64,
    #[prost(enumeration="channel_handle::ChannelType", tag="2")]
    pub r#type: i32,
}
pub mod channel_handle {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum ChannelType {
        /// Invalid primitive type to serve as default.
        Invalid = 0,
        /// A channel for sending data between devices.
        DeviceToDevice = 1,
        /// A channel for sending data from the device to the host. Can only be used
        /// with a Send operation.
        DeviceToHost = 2,
        /// A channel for sending data from the host to the device. Can only be used
        /// with a Recv operation.
        HostToDevice = 3,
    }
}
/// DeviceAssignmentProto is a serialized form of DeviceAssignment class, which
/// represents the device ids assigned to a set of replicated computations.
/// See xla::DeviceAssignment class comment for more details.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeviceAssignmentProto {
    #[prost(int32, tag="1")]
    pub replica_count: i32,
    #[prost(int32, tag="2")]
    pub computation_count: i32,
    #[prost(message, repeated, tag="3")]
    pub computation_devices: ::std::vec::Vec<device_assignment_proto::ComputationDevice>,
}
pub mod device_assignment_proto {
    /// Each logical computation runs on replica_count physical devices.
    /// ComputationDevice represents the device ids assinged to the replicas.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ComputationDevice {
        #[prost(int32, repeated, tag="1")]
        pub replica_device_ids: ::std::vec::Vec<i32>,
    }
}
/// Literals are used when the server and client need to exchange materialized
/// data / results. Literals are also used to describe constants used in
/// computations.
///
/// Transfers to/from the client are encoded in literal form, and the structure
/// of the repeated fields is implied by the shape.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LiteralProto {
    #[prost(message, optional, tag="1")]
    pub shape: ::std::option::Option<ShapeProto>,
    #[prost(bool, repeated, tag="2")]
    pub preds: ::std::vec::Vec<bool>,
    #[prost(bytes, tag="15")]
    pub s8s: std::vec::Vec<u8>,
    #[prost(bytes, tag="3")]
    pub u8s: std::vec::Vec<u8>,
    #[prost(int32, repeated, tag="4")]
    pub s32s: ::std::vec::Vec<i32>,
    #[prost(int64, repeated, tag="5")]
    pub s64s: ::std::vec::Vec<i64>,
    #[prost(uint32, repeated, tag="6")]
    pub u32s: ::std::vec::Vec<u32>,
    #[prost(uint64, repeated, tag="7")]
    pub u64s: ::std::vec::Vec<u64>,
    #[prost(float, repeated, tag="8")]
    pub f32s: ::std::vec::Vec<f32>,
    #[prost(double, repeated, tag="9")]
    pub f64s: ::std::vec::Vec<f64>,
    /// Stored as interleaved real, imag floats.
    #[prost(float, repeated, tag="12")]
    pub c64s: ::std::vec::Vec<f32>,
    /// Stored as interleaved real, imag doubles.
    #[prost(double, repeated, tag="18")]
    pub c128s: ::std::vec::Vec<f64>,
    #[prost(message, repeated, tag="10")]
    pub tuple_literals: ::std::vec::Vec<LiteralProto>,
    /// The F16s, BF16s, U16s and S16s are encoded in little endian byte order
    #[prost(bytes, tag="11")]
    pub f16s: std::vec::Vec<u8>,
    #[prost(bytes, tag="13")]
    pub bf16s: std::vec::Vec<u8>,
    #[prost(bytes, tag="16")]
    pub u16s: std::vec::Vec<u8>,
    #[prost(bytes, tag="17")]
    pub s16s: std::vec::Vec<u8>,
    /// Next = 19
    #[prost(int64, repeated, tag="14")]
    pub sparse_indices: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowDimension {
    /// The size of the window in this dimension. For a rectangle, this would be
    /// the width or height.
    #[prost(int64, tag="1")]
    pub size: i64,
    /// The stride at which the window moves across the base area in this
    /// dimension. In other words, this is the spacing between different
    /// positions of the window in this dimension.
    #[prost(int64, tag="2")]
    pub stride: i64,
    /// If positive, means the amount of padding to add to the base area at the low
    /// end of this dimension; if negative, its negative means the number of
    /// elements removed from the low end of this dimension. For example, in the
    /// horizontal dimension of a rectangle, this would be the number of padding
    /// values to pad on the left, given that indices increase when going right.
    /// The actual padding value depends upon the context. Convolution pads with
    /// zeros. ReduceWindow and SelectAndScatter pads with the reduce function's
    /// init value.
    #[prost(int64, tag="3")]
    pub padding_low: i64,
    /// As padding_low, but on the high end of this dimension. For example, in the
    /// horizontal dimension of a rectangle, this would be the number of values to
    /// pad on the right, given that indices increase when going right.
    #[prost(int64, tag="4")]
    pub padding_high: i64,
    /// Dilation factor of the sliding window in this dimension. A dilation factor
    /// of 1 means no dilation. window_dilation - 1 no-op entries ("holes") are
    /// implicitly placed between each kernel element. This value may not be less
    /// than 1. See documentation for convolution.
    #[prost(int64, tag="5")]
    pub window_dilation: i64,
    /// Dilation factor of the base area in this dimension. A dilation factor of 1
    /// means no dilation. base_dilation - 1 no-op entries ("holes") are implicitly
    /// placed between each base area element. This value may not be less than 1.
    /// See documentation for convolution.
    #[prost(int64, tag="6")]
    pub base_dilation: i64,
    /// Window reversal means that this dimension was logically reversed before the
    /// operation.
    #[prost(bool, tag="7")]
    pub window_reversal: bool,
}
/// Describes the windowing in an operation such as convolution.
///
/// The window is moved across a base area and for each position of the
/// window a computation is performed. The field below describes the
/// window and the movement of the window across a base area.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Window {
    #[prost(message, repeated, tag="1")]
    pub dimensions: ::std::vec::Vec<WindowDimension>,
}
/// Describes the dimension numbers for a gather operation.
///
/// See https://www.tensorflow.org/performance/xla/operation_semantics#gather for
/// more details.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GatherDimensionNumbers {
    /// "Window indices" is a term for a set of indices that index into the
    /// interior of a dynamic-slice from the input tensor, the starting indices for
    /// which were computed from output_gather_dims (see the operation semantic for
    /// how this is defined) and the start_indices tensor.
    ///
    /// The window indices for a specific output index Out is computed as:
    ///
    ///  i = 0
    ///  for (k : [0, input_tensor_shape.rank))
    ///    window_indices[k] =
    ///      if k in collapsed_slice_dims
    ///      then 0
    ///      else Out[offset_dims[i++]]
    #[prost(int64, repeated, tag="1")]
    pub offset_dims: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="2")]
    pub collapsed_slice_dims: ::std::vec::Vec<i64>,
    /// This is interpreted as a map from i to start_index_map[i]. It
    /// transforms the gather index looked up from the start_indices tensor into
    /// the starting index in the input space.
    #[prost(int64, repeated, tag="3")]
    pub start_index_map: ::std::vec::Vec<i64>,
    /// The dimension in the start_indices input that contains the starting
    /// indices.
    #[prost(int64, tag="4")]
    pub index_vector_dim: i64,
}
/// Describes the dimension numbers for a scatter operation.
///
/// All the fields are similar to the corresponding fields in
/// GatherDimensionNumbers. Differences are noted below.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ScatterDimensionNumbers {
    /// The set of dimensions in the updates shape that are window dimensions.
    #[prost(int64, repeated, tag="1")]
    pub update_window_dims: ::std::vec::Vec<i64>,
    /// The set of window dimensions that must be inserted into the updates shape.
    #[prost(int64, repeated, tag="2")]
    pub inserted_window_dims: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="3")]
    pub scatter_dims_to_operand_dims: ::std::vec::Vec<i64>,
    #[prost(int64, tag="4")]
    pub index_vector_dim: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConvolutionDimensionNumbers {
    /// The number of the dimension that represents batch in the input.
    #[prost(int64, tag="7")]
    pub input_batch_dimension: i64,
    /// The number of the dimension that represents features in the input.
    #[prost(int64, tag="8")]
    pub input_feature_dimension: i64,
    /// The dimension numbers for the spatial dimensions that the window
    /// moves through in the input.
    #[prost(int64, repeated, tag="11")]
    pub input_spatial_dimensions: ::std::vec::Vec<i64>,
    /// The number of the dimension that represents input features in the
    /// convolutional kernel (rhs).
    #[prost(int64, tag="3")]
    pub kernel_input_feature_dimension: i64,
    /// The number of the dimension that represents output features in
    /// the convolutional kernel (rhs).
    #[prost(int64, tag="4")]
    pub kernel_output_feature_dimension: i64,
    /// The dimension numbers for the spatial dimensions that the window
    /// moves through in the kernel (rhs). window.strides(0) is the
    /// stride in the kernel_spatial_dimensions(0) dimension.
    #[prost(int64, repeated, tag="6")]
    pub kernel_spatial_dimensions: ::std::vec::Vec<i64>,
    /// The number of the dimension that represents batch in the output.
    #[prost(int64, tag="9")]
    pub output_batch_dimension: i64,
    /// The number of the dimension that represents features in the output.
    #[prost(int64, tag="10")]
    pub output_feature_dimension: i64,
    /// The dimension numbers for the spatial dimensions that the window
    /// moves through in the output.
    #[prost(int64, repeated, tag="12")]
    pub output_spatial_dimensions: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DotDimensionNumbers {
    /// The dimension numbers that represent the 'lhs' contracting dimensions.
    #[prost(int64, repeated, tag="1")]
    pub lhs_contracting_dimensions: ::std::vec::Vec<i64>,
    /// The dimension numbers that represent the 'rhs' contracting dimensions.
    #[prost(int64, repeated, tag="2")]
    pub rhs_contracting_dimensions: ::std::vec::Vec<i64>,
    /// The dimension numbers that represent the 'lhs' batch dimensions.
    #[prost(int64, repeated, tag="3")]
    pub lhs_batch_dimensions: ::std::vec::Vec<i64>,
    /// The dimension numbers that represent the 'rhs' batch dimensions.
    #[prost(int64, repeated, tag="4")]
    pub rhs_batch_dimensions: ::std::vec::Vec<i64>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TriangularSolveOptions {
    /// If true, solves ax = b. If false, solves xa = b.
    #[prost(bool, tag="1")]
    pub left_side: bool,
    /// If true, 'a' is lower triangular. If false, 'a' is upper triangular.
    #[prost(bool, tag="2")]
    pub lower: bool,
    /// If true, the diagonal elements of 'a' are assumed to be 1 and not accessed.
    #[prost(bool, tag="3")]
    pub unit_diagonal: bool,
    #[prost(enumeration="triangular_solve_options::Transpose", tag="4")]
    pub transpose_a: i32,
}
pub mod triangular_solve_options {
    /// Should we transpose or use the adjoint of 'a'?
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Transpose {
        Invalid = 0,
        /// Don't transpose 'a'.
        NoTranspose = 1,
        /// Transpose 'a'.
        Transpose = 2,
        /// Complex conjugate and transpose 'a'.
        Adjoint = 3,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CholeskyOptions {
    /// If true, uses the lower triangle of `a`. If false, uses the upper triangle
    /// of `a`.
    #[prost(bool, tag="1")]
    pub lower: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OpSharding {
    #[prost(enumeration="op_sharding::Type", tag="1")]
    pub r#type: i32,
    /// The shape of the sharded tile.
    #[prost(message, optional, tag="2")]
    pub tile_shape: ::std::option::Option<ShapeProto>,
    /// The shape of the tile assignment tensor - this must be the same rank as
    /// tile_shape and the product of its dimensions must equal
    /// tile_assignment_devices.size().
    #[prost(int64, repeated, tag="3")]
    pub tile_assignment_dimensions: ::std::vec::Vec<i64>,
    /// Flattened list of device IDs. The order of flattening is the same as used
    /// by IndexUtil::MultiToLinearIndex(tile_assignment_shape).
    #[prost(int64, repeated, tag="4")]
    pub tile_assignment_devices: ::std::vec::Vec<i64>,
    /// If type == TUPLE, the sub-shardings, one per leaf node in the tuple shape,
    /// in pre-order. The tuple shape could be nested; here we store just a
    /// flattened list of all leaves in the tuple shape. Note that the tuple shape
    /// is not stored here; shardings do not store the shapes to which they are
    /// applied, this is inferred from the instruction this sharding gets attached
    /// to.
    #[prost(message, repeated, tag="5")]
    pub tuple_shardings: ::std::vec::Vec<OpSharding>,
}
pub mod op_sharding {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Type {
        /// This sharding is replicated across all devices (implies maximal,
        /// all other fields are unused).
        Replicated = 0,
        /// This sharding is maximal - one device runs the entire operation.
        Maximal = 1,
        /// This sharding is a tuple - only the tuple_shardings field is valid.
        Tuple = 2,
        /// None of the above; tile_shape and tile_assignment are both used.
        Other = 3,
    }
}
/// Describes the replica groups in a cross replica op (e.g., all-reduce and
/// all-to-all).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReplicaGroup {
    /// The ids of the replicas that belongs to the same group. The ordering of the
    /// ids matters in some ops (e.g., all-to-all).
    #[prost(int64, repeated, tag="1")]
    pub replica_ids: ::std::vec::Vec<i64>,
}
/// Describes the source target pair in the collective permute op.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SourceTarget {
    #[prost(int64, tag="1")]
    pub source: i64,
    #[prost(int64, tag="2")]
    pub target: i64,
}
/// Used to indicate the precision configuration. It has backend specific
/// meaning.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrecisionConfig {
    #[prost(enumeration="precision_config::Precision", repeated, tag="1")]
    pub operand_precision: ::std::vec::Vec<i32>,
}
pub mod precision_config {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Precision {
        Default = 0,
        High = 1,
        Highest = 2,
    }
}
/// Describes whether all data-parallelism replicas will receive the same
/// parameter data at each buffer.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ParameterReplication {
    /// A list of boolean values for the flattened leaf buffers. Each value
    /// indicates whether the corresponding leaf buffer is replicated.
    ///
    /// If this field is empty, it means no buffer is replicated. Otherwise, the
    /// number of elements in this field must match the number of leaf buffers in
    /// the HLO instruction's shape.
    #[prost(bool, repeated, tag="1")]
    pub replicated_at_leaf_buffers: ::std::vec::Vec<bool>,
}
/// A backend-config for kWhile loops that stores the loop's trip count, if it is
/// known.
///
/// This is useful for backends that can implement a `for i in 0..N` loop more
/// efficiently than a `while` loop.  For example, on GPUs, we can implement a
/// `for i in 0..N` loop by enqueueing the kernels for the loop body N times,
/// whereas implementing a `while` loop requires a host-device sync on each
/// iteration.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WhileLoopBackendConfig {
    /// This indirection lets us distinguish between known-trip-count == 0 and
    /// unknown-trip-count.
    #[prost(message, optional, tag="1")]
    pub known_trip_count: ::std::option::Option<while_loop_backend_config::KnownTripCount>,
}
pub mod while_loop_backend_config {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct KnownTripCount {
        #[prost(int64, tag="1")]
        pub n: i64,
    }
}
/// Primitive types are the individual values that can be held in rectangular
/// multidimensional arrays. A description of the rectangular multidimensional
/// array dimensions / primitive type is given by Shape, below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum PrimitiveType {
    /// Invalid primitive type to serve as default.
    Invalid = 0,
    /// Predicates are two-state booleans.
    Pred = 1,
    /// Signed integral values of fixed width.
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    /// Unsigned integral values of fixed width.
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    /// Floating-point values of fixed width.
    ///
    /// Note: if f16s are not natively supported on the device, they will be
    /// converted to f16 from f32 at arbirary points in the computation.
    F16 = 10,
    F32 = 11,
    /// Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
    /// floating-point format, but uses 1 bit for the sign, 8 bits for the exponent
    /// and 7 bits for the mantissa.
    Bf16 = 16,
    F64 = 12,
    /// Complex values of fixed width.
    ///
    /// Paired F32 (real, imag), as in std::complex<float>.
    C64 = 15,
    /// Paired F64 (real, imag), as in std::complex<double>.
    C128 = 18,
    /// A tuple is a polymorphic sequence; e.g. a shape that holds different
    /// sub-shapes. They are used for things like returning multiple values from a
    /// computation; e.g. a computation that returns weights and biases may have a
    /// signature that results in a tuple like (f32[784x2000], f32[2000])
    ///
    /// If a shape proto has the tuple element type, it may not have any entries
    /// in the dimensions field.
    Tuple = 13,
    /// An opaque type used for passing context-specific data to a custom
    /// operation. Shapes of this primitive type will have empty dimensions and
    /// tuple_shapes fields.
    ///
    /// (OPAQUE would be a better name for this identifier, but that conflicts with
    /// a macro defined in windows.h.)
    OpaqueType = 14,
    /// A token type threaded between side-effecting operations. Shapes of this
    /// primitive type will have empty dimensions and tuple_shapes fields.
    Token = 17,
}
/// A format specifies the method used by a layout to store an array in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Format {
    /// TODO(b/120869032): Rename this to FORMAT_NONE or something else which
    /// better corresponds to its meaning.
    InvalidFormat = 0,
    /// The default layout, with exactly one storage location per element.
    Dense = 1,
    /// A sparsely encoded layout, providing only the index/value pairs of non-zero
    /// elements.
    Sparse = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum FftType {
    /// Forward FFT; complex in, complex out.
    Fft = 0,
    /// Inverse FFT; complex in, complex out.
    Ifft = 1,
    /// Forward real FFT; real in, fft_length / 2 + 1 complex out
    Rfft = 2,
    /// Inverse real FFT; fft_length / 2 + 1 complex in,
    Irfft = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum RandomDistribution {
    RngInvalid = 0,
    /// Creates a uniform-distribution-generated random number on the semi-open
    /// interval [parameter[0], parameter[1]).
    RngUniform = 1,
    /// Creates a normal-distribution-generated random number with mean
    /// parameter[0] and standard deviation parameter[1].
    RngNormal = 2,
}
/// Serialization of HloInstruction.
/// Next ID: 68
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloInstructionProto {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(string, tag="2")]
    pub opcode: std::string::String,
    #[prost(message, optional, tag="3")]
    pub shape: ::std::option::Option<ShapeProto>,
    #[prost(message, optional, tag="7")]
    pub metadata: ::std::option::Option<OpMetadata>,
    /// Literal, only present for kConstant.
    #[prost(message, optional, tag="8")]
    pub literal: ::std::option::Option<LiteralProto>,
    /// Parameter number is only present for kParameter.
    #[prost(int64, tag="9")]
    pub parameter_number: i64,
    /// Fusion state, only present for kFusion.
    #[prost(string, tag="11")]
    pub fusion_kind: std::string::String,
    /// Index for kGetTupleElement.
    #[prost(int64, tag="13")]
    pub tuple_index: i64,
    /// Dimensions present for some operations that require reshaping or
    /// broadcasting, including Reshape, Reduce, ReduceWindow, and Reverse.
    #[prost(int64, repeated, tag="14")]
    pub dimensions: ::std::vec::Vec<i64>,
    /// Describes the window in a windowed operation such as convolution.
    #[prost(message, optional, tag="15")]
    pub window: ::std::option::Option<Window>,
    /// Describes the dimension numbers used for a convolution.
    #[prost(message, optional, tag="16")]
    pub convolution_dimension_numbers: ::std::option::Option<ConvolutionDimensionNumbers>,
    /// The number of feature groups. Used for a convolution. Must be a divisor of
    /// the input feature dimension and output feature dimension. If not specified,
    /// it will use a default value of 1.
    #[prost(int64, tag="50")]
    pub feature_group_count: i64,
    #[prost(int64, tag="58")]
    pub batch_group_count: i64,
    #[prost(message, repeated, tag="17")]
    pub slice_dimensions: ::std::vec::Vec<hlo_instruction_proto::SliceDimensions>,
    /// The bit sizes for a reduce-precision operation.
    #[prost(int32, tag="18")]
    pub exponent_bits: i32,
    #[prost(int32, tag="19")]
    pub mantissa_bits: i32,
    /// Describes the [start, start + size) range size for a dynamic slice
    /// ('start' is specified dynamically in the second operand of the operation).
    #[prost(int64, repeated, tag="20")]
    pub dynamic_slice_sizes: ::std::vec::Vec<i64>,
    /// The padding configuration that describes the edge padding and interior
    /// padding of this pad instruction. Only set for pad instructions.
    #[prost(message, optional, tag="21")]
    pub padding_config: ::std::option::Option<PaddingConfig>,
    /// Outfeed configuration information, only present for kOutfeed.
    #[prost(bytes, tag="22")]
    pub outfeed_config: std::vec::Vec<u8>,
    /// The distribution requested for random number generation.
    /// Only present for kRng.
    #[prost(enumeration="RandomDistribution", tag="23")]
    pub distribution: i32,
    /// A small float number added to the variance to avoid divide-by-zero error.
    /// Only present for kBatchNormTraining.
    #[prost(float, tag="24")]
    pub epsilon: f32,
    /// An integer value representing the index of the feature dimension.
    /// Only present for kBatchNormTraining.
    #[prost(int64, tag="25")]
    pub feature_index: i64,
    /// Represents a unique identifier for each Send/Recv instruction pair or
    /// optionally for collective instructions (AllReduce, CollectivePermute,
    /// AllToAll). Non-positive channel_id is equivalent to no channel id.
    #[prost(int64, tag="26")]
    pub channel_id: i64,
    /// The string representation of the infeed configuration.
    #[prost(bytes, tag="27")]
    pub infeed_config: std::vec::Vec<u8>,
    /// Name of a external target (eg, global symbol) to call, only present for
    /// kCustomCall.
    #[prost(string, tag="28")]
    pub custom_call_target: std::string::String,
    /// Shape of outfeed request.
    #[prost(message, optional, tag="29")]
    pub outfeed_shape: ::std::option::Option<ShapeProto>,
    /// Describes the dimension numbers used for a dot operation
    #[prost(message, optional, tag="30")]
    pub dot_dimension_numbers: ::std::option::Option<DotDimensionNumbers>,
    /// FFT type (FFT, IFFT, etc).
    #[prost(enumeration="FftType", tag="31")]
    pub fft_type: i32,
    /// FFT length.
    #[prost(int64, repeated, tag="32")]
    pub fft_length: ::std::vec::Vec<i64>,
    /// Comparison direction only used for kCompare.
    #[prost(string, tag="63")]
    pub comparison_direction: std::string::String,
    /// Gather dimension numbers.
    #[prost(message, optional, tag="33")]
    pub gather_dimension_numbers: ::std::option::Option<GatherDimensionNumbers>,
    #[prost(int64, repeated, tag="34")]
    pub gather_slice_sizes: ::std::vec::Vec<i64>,
    /// Compute Host.
    #[prost(string, tag="41")]
    pub channel_name: std::string::String,
    #[prost(int64, tag="42")]
    pub cost_estimate_ns: i64,
    /// The id of this instruction.
    #[prost(int64, tag="35")]
    pub id: i64,
    #[prost(int64, repeated, tag="36")]
    pub operand_ids: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="37")]
    pub control_predecessor_ids: ::std::vec::Vec<i64>,
    #[prost(int64, repeated, tag="38")]
    pub called_computation_ids: ::std::vec::Vec<i64>,
    #[prost(message, optional, tag="40")]
    pub sharding: ::std::option::Option<OpSharding>,
    /// Backend configuration for the instruction. Has backend-specific meaning.
    #[prost(string, tag="43")]
    pub backend_config: std::string::String,
    /// Cross replica op fields.
    #[prost(message, repeated, tag="49")]
    pub replica_groups: ::std::vec::Vec<ReplicaGroup>,
    /// Deprecated, but keeping it for backward compatibility. Use channel_id.
    /// Non-positive all_reduce_id is equivalent to no all_reduce_id.
    #[prost(int64, tag="45")]
    pub all_reduce_id: i64,
    /// Whether this Send/Recv instruction transfers data to/from the host. Only
    /// present for Send and Recv instructions and their SendDone and RecvDone
    /// partners.
    #[prost(bool, tag="47")]
    pub is_host_transfer: bool,
    /// Whether this Sort instruction should be stable.
    #[prost(bool, tag="60")]
    pub is_stable: bool,
    #[prost(message, optional, tag="48")]
    pub scatter_dimension_numbers: ::std::option::Option<ScatterDimensionNumbers>,
    /// Precision configuration for the instruction. Has backend-specific meaning.
    #[prost(message, optional, tag="51")]
    pub precision_config: ::std::option::Option<PrecisionConfig>,
    /// Collective permute field.
    #[prost(message, repeated, tag="52")]
    pub source_target_pairs: ::std::vec::Vec<SourceTarget>,
    /// Sharding for kDomain instructions.
    #[prost(message, optional, tag="54")]
    pub domain_entry_sharding: ::std::option::Option<OpSharding>,
    #[prost(message, optional, tag="55")]
    pub domain_exit_sharding: ::std::option::Option<OpSharding>,
    /// For custom call this indicates that the layouts are constrained. If
    /// constrain_layout is true then the 'shape' field must contain a layout, and
    /// 'operand_shapes_with_layout' must contain a shape with layout for each
    /// operand.
    #[prost(bool, tag="56")]
    pub constrain_layout: bool,
    #[prost(message, repeated, tag="57")]
    pub operand_shapes_with_layout: ::std::vec::Vec<ShapeProto>,
    /// Options for TriangularSolve
    #[prost(message, optional, tag="59")]
    pub triangular_solve_options: ::std::option::Option<TriangularSolveOptions>,
    /// Options for Cholesky
    #[prost(message, optional, tag="62")]
    pub cholesky_options: ::std::option::Option<CholeskyOptions>,
    /// Describes how parameters behave with regards to replicas.
    #[prost(message, optional, tag="61")]
    pub parameter_replication: ::std::option::Option<ParameterReplication>,
    /// If set, the given instruction is run in parallel on e.g. multiple CPU
    /// cores.  The outermost dimension gets split up into
    /// outer_dimension_partitions[0] pieces, the next-outermost dim gets split
    /// into outer_dimension_partitions[1] pieces, etc.
    ///
    /// It's illegal to partition a dimension into more shards than there are
    /// elements in that dimension.
    #[prost(int64, repeated, tag="64")]
    pub outer_dimension_partitions: ::std::vec::Vec<i64>,
    /// Whether the kCustomCall instruction has side-effects, only present for
    /// kCustomCall.
    #[prost(bool, tag="65")]
    pub custom_call_has_side_effect: bool,
    /// The delta value for kRngGetAndUpdateState.
    #[prost(int64, tag="66")]
    pub delta: i64,
    /// Specifies if the gather/scatter indices are guaranteed to be sorted by the
    /// caller.
    #[prost(bool, tag="67")]
    pub indices_are_sorted: bool,
}
pub mod hlo_instruction_proto {
    /// Describes the [begin, end) index range and stride for slices.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SliceDimensions {
        #[prost(int64, tag="1")]
        pub start: i64,
        #[prost(int64, tag="2")]
        pub limit: i64,
        #[prost(int64, tag="3")]
        pub stride: i64,
    }
}
/// Serialization of HloComputation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloComputationProto {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The array of instructions is always in a valid dependency order, where
    /// operands appear before their users.
    #[prost(message, repeated, tag="2")]
    pub instructions: ::std::vec::Vec<HloInstructionProto>,
    // The program shape (with layout) of this computation.

    #[prost(message, optional, tag="4")]
    pub program_shape: ::std::option::Option<ProgramShapeProto>,
    /// The id of this computation.
    #[prost(int64, tag="5")]
    pub id: i64,
    /// The id of the root of the computation.
    #[prost(int64, tag="6")]
    pub root_id: i64,
}
/// Serialization of an HLO schedule. An HLO schedule contains a total order of
/// instructions for each non-fusion computation in the module.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloScheduleProto {
    /// Map from computation id to sequence.
    #[prost(map="int64, message", tag="1")]
    pub sequences: ::std::collections::HashMap<i64, hlo_schedule_proto::InstructionSequence>,
}
pub mod hlo_schedule_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InstructionSequence {
        #[prost(int64, repeated, tag="1")]
        pub instruction_ids: ::std::vec::Vec<i64>,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloInputOutputAliasProto {
    #[prost(message, repeated, tag="1")]
    pub entries: ::std::vec::Vec<hlo_input_output_alias_proto::AliasEntryProto>,
}
pub mod hlo_input_output_alias_proto {
    /// The following proto describes a pair of aliased an input
    /// (described by parameter number and a ShapeIndex of the parameter)
    /// and an output (described by a ShapeIndex of the root
    /// instruction). For example:
    ///
    /// entry = {
    ///  output_shape_index={1},
    ///  parameter_number=0,
    ///  parameter_shape_index={1, 2},
    /// }
    ///
    /// This entry indicates that the first paremter's {1, 2} element is
    /// aliased with the {1} element of the root instruction.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AliasEntryProto {
        /// ShapeIndex of the root hlo.
        #[prost(int64, repeated, tag="1")]
        pub output_shape_index: ::std::vec::Vec<i64>,
        /// Number of the parameter in entry computation.
        #[prost(int64, tag="2")]
        pub parameter_number: i64,
        /// ShapeIndex of the parameter instruction.
        #[prost(int64, repeated, tag="3")]
        pub parameter_shape_index: ::std::vec::Vec<i64>,
        /// The kind of alias to be setup.
        #[prost(enumeration="Kind", tag="4")]
        pub kind: i32,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Kind {
        /// Define a UNDEFINED_ALIAS equal to zero to get around the default-0 proto3
        /// behavior and missing has_*() APIs.
        UndefinedAlias = 0,
        /// An alias setup by the user as must alias. A use setting USER_ALIAS is
        /// expecting the designed output to be dropped over the given input
        /// parameter number+index.
        UserAlias = 1,
        /// An alias setup by the compiler as part of its optimizations.
        SystemAlias = 2,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DynamicParameterBindingProto {
    #[prost(message, repeated, tag="1")]
    pub entries: ::std::vec::Vec<dynamic_parameter_binding_proto::Binding>,
}
pub mod dynamic_parameter_binding_proto {
    /// A list of bindings which indicates that the `target_dim_num` in
    /// the subshape `target_param_index` of parameter `target_param_num`
    /// is a dynamic dimension and its real dynamic size is represented
    /// by `dynamic_param_index` in parameter `dynamic_param_num`.
    ///
    /// As an example, imagine we have a program:
    ///
    /// ENTRY main {
    ///   a = f32[] parameter(0)
    ///   b = f32[10] parameter(1)
    ///   ROOT root = (f32[], f32[10]) tuple(%a, %b)
    /// }
    ///
    /// Let's say 'b' (param index 1) is a dynamic shape whose input has
    /// an upperbound of 10 and real size is determined at runtime.'a'
    /// represents the real size of b's first dimension.
    ///
    /// In this case, the fields are set in the following way:
    /// dynamic_param_num = 1
    /// dynamic_param_index = {}
    /// target_param_num = 0
    /// target_param_index = {}
    /// target_param_dim = 0
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Binding {
        #[prost(int64, tag="1")]
        pub dynamic_param_num: i64,
        #[prost(int64, repeated, tag="2")]
        pub dynamic_param_index: ::std::vec::Vec<i64>,
        #[prost(int64, tag="3")]
        pub target_param_num: i64,
        #[prost(int64, repeated, tag="4")]
        pub target_param_index: ::std::vec::Vec<i64>,
        #[prost(int64, tag="5")]
        pub target_param_dim_num: i64,
    }
}
/// Serialization of HloModule.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloModuleProto {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(string, tag="2")]
    pub entry_computation_name: std::string::String,
    #[prost(int64, tag="6")]
    pub entry_computation_id: i64,
    /// The array of computations is always in a valid dependency order, where
    /// callees appear before their callers.
    #[prost(message, repeated, tag="3")]
    pub computations: ::std::vec::Vec<HloComputationProto>,
    /// The host program shape (with layout) of the entry computation.
    #[prost(message, optional, tag="4")]
    pub host_program_shape: ::std::option::Option<ProgramShapeProto>,
    /// The id of this module.
    #[prost(int64, tag="5")]
    pub id: i64,
    /// The schedule for this module.
    #[prost(message, optional, tag="7")]
    pub schedule: ::std::option::Option<HloScheduleProto>,
    /// Describes alias information between inputs and outputs.
    #[prost(message, optional, tag="8")]
    pub input_output_alias: ::std::option::Option<HloInputOutputAliasProto>,
    #[prost(message, optional, tag="9")]
    pub dynamic_parameter_binding: ::std::option::Option<DynamicParameterBindingProto>,
}
/// Serialization of LogicalBuffer.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LogicalBufferProto {
    #[prost(int64, tag="1")]
    pub id: i64,
    #[prost(int64, tag="2")]
    pub size: i64,
    /// The location where the buffer is defined.
    #[prost(message, optional, tag="3")]
    pub defined_at: ::std::option::Option<logical_buffer_proto::Location>,
    #[prost(int64, tag="4")]
    pub color: i64,
}
pub mod logical_buffer_proto {
    /// Location represents an instruction and its shape index, which uniquely
    /// identifies a point where a buffer is needed.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Location {
        /// NOTE: module_name isn't necessary, since all LogicalBuffers are
        /// associated with a single HloModule.
        #[prost(string, tag="1")]
        pub computation_name: std::string::String,
        #[prost(string, tag="2")]
        pub instruction_name: std::string::String,
        #[prost(int64, repeated, tag="3")]
        pub shape_index: ::std::vec::Vec<i64>,
    }
}
/// Serialization of BufferAllocation.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BufferAllocationProto {
    #[prost(int64, tag="1")]
    pub index: i64,
    #[prost(int64, tag="2")]
    pub size: i64,
    #[prost(bool, tag="3")]
    pub is_thread_local: bool,
    #[prost(bool, tag="11")]
    pub is_tuple: bool,
    #[prost(bool, tag="5")]
    pub is_entry_computation_parameter: bool,
    #[prost(bool, tag="12")]
    pub is_constant: bool,
    #[prost(int64, tag="6")]
    pub parameter_number: i64,
    #[prost(int64, repeated, tag="10")]
    pub parameter_shape_index: ::std::vec::Vec<i64>,
    #[prost(bool, tag="7")]
    pub maybe_live_out: bool,
    #[prost(int64, tag="8")]
    pub color: i64,
    #[prost(message, repeated, tag="9")]
    pub assigned: ::std::vec::Vec<buffer_allocation_proto::Assigned>,
}
pub mod buffer_allocation_proto {
    /// Assigned represents a single LogicalBuffer that is assigned to this
    /// BufferAllocation.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Assigned {
        #[prost(int64, tag="1")]
        pub logical_buffer_id: i64,
        #[prost(int64, tag="2")]
        pub offset: i64,
        #[prost(int64, tag="3")]
        pub size: i64,
    }
}
/// A trace of a HeapSimulator run.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HeapSimulatorTrace {
    #[prost(message, repeated, tag="1")]
    pub events: ::std::vec::Vec<heap_simulator_trace::Event>,
    #[prost(bool, tag="2")]
    pub whole_module_simulation: bool,
}
pub mod heap_simulator_trace {
    /// The trace includes a list of events, where each event describes one action
    /// performed by the heap simulator.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Event {
        #[prost(enumeration="event::Kind", tag="1")]
        pub kind: i32,
        /// The id of the LogicalBuffer that the event applies to.
        #[prost(int64, tag="2")]
        pub buffer_id: i64,
        /// The HloInstruction that the simulation was processing that caused this
        /// event to occur, identified by its computation and instruction name. E.g.
        /// buffers defined by instruction A are allocated when processing A.
        #[prost(string, tag="3")]
        pub computation_name: std::string::String,
        #[prost(string, tag="4")]
        pub instruction_name: std::string::String,
        /// The id of the canonical LogicalBuffer that the buffer shares with. Only
        /// set for SHARE_WITH events.
        #[prost(int64, tag="5")]
        pub share_with_canonical_id: i64,
    }
    pub mod event {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
        #[repr(i32)]
        pub enum Kind {
            /// A memory region was allocated for the buffer.
            Alloc = 0,
            /// A memory region was freed for the buffer.
            Free = 1,
            /// A buffer was shared with another (canonical) buffer. This is similar to
            /// ALLOC, except that instead of allocating a new region of memory, the
            /// memory region of the canonical buffer is directly re-used. Multiple
            /// buffers may share with the same canonical buffer. The lifetime of the
            /// canonical buffer is extended to the union of all lifetimes.
            ShareWith = 2,
        }
    }
}
/// An abstraction representing a set of HLO module built to run concurrently
/// across different devices.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloModuleGroupProto {
    #[prost(string, tag="1")]
    pub name: std::string::String,
    #[prost(message, repeated, tag="2")]
    pub hlo_modules: ::std::vec::Vec<HloModuleProto>,
}
/// Serialization of BufferAssignment.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BufferAssignmentProto {
    #[prost(message, repeated, tag="1")]
    pub logical_buffers: ::std::vec::Vec<LogicalBufferProto>,
    #[prost(message, repeated, tag="2")]
    pub buffer_aliases: ::std::vec::Vec<buffer_assignment_proto::BufferAlias>,
    #[prost(message, repeated, tag="3")]
    pub buffer_allocations: ::std::vec::Vec<BufferAllocationProto>,
    #[prost(message, repeated, tag="4")]
    pub heap_simulator_traces: ::std::vec::Vec<HeapSimulatorTrace>,
}
pub mod buffer_assignment_proto {
    /// Alias represents a source LogicalBuffer, and the buffer location that
    /// aliases it.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct BufferAlias {
        #[prost(int64, tag="1")]
        pub source_buffer_id: i64,
        #[prost(message, optional, tag="2")]
        pub location: ::std::option::Option<super::logical_buffer_proto::Location>,
    }
}
/// Grouping message that contains all of the information above.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloProto {
    #[prost(message, optional, tag="1")]
    pub hlo_module: ::std::option::Option<HloModuleProto>,
    #[prost(message, optional, tag="3")]
    pub buffer_assignment: ::std::option::Option<BufferAssignmentProto>,
}
/// Encapsulates HloProto together with the arguments, result, and
/// execution_platform. This message is used for purposes such as
/// analysis/replay/file-storage.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloSnapshot {
    /// The hlo graph.
    #[prost(message, optional, tag="1")]
    pub hlo: ::std::option::Option<HloProto>,
    /// The arguments passed to the graph.
    #[prost(message, repeated, tag="2")]
    pub arguments: ::std::vec::Vec<LiteralProto>,
    /// The result of the graph.
    #[prost(message, optional, tag="3")]
    pub result: ::std::option::Option<LiteralProto>,
    /// The name of the platform used to run the graph.
    #[prost(string, tag="4")]
    pub execution_platform: std::string::String,
}
/// Describes how to pretty-print a profile counter array gathered for a specific
/// HloModule.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloProfilePrinterData {
    /// HloComputationInfos for every HloComputation in the HloModule.
    #[prost(message, repeated, tag="1")]
    pub computation_infos: ::std::vec::Vec<hlo_profile_printer_data::HloComputationInfo>,
    /// The size of the profile counters array we will pretty-print.
    #[prost(int64, tag="2")]
    pub profile_counters_size: i64,
    /// Maps extra metric name to the index into the profile counters array.
    #[prost(map="string, int64", tag="3")]
    pub extra_metrics: ::std::collections::HashMap<std::string::String, i64>,
    /// Name of the entry computation.
    #[prost(string, tag="4")]
    pub entry_computation: std::string::String,
}
pub mod hlo_profile_printer_data {
    /// Pretty-printer information about an HloInstruction.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HloInstructionInfo {
        #[prost(string, tag="1")]
        pub long_name: std::string::String,
        #[prost(string, tag="2")]
        pub short_name: std::string::String,
        #[prost(string, tag="3")]
        pub category: std::string::String,
        /// Metrics computed by HloCostAnalysis.
        #[prost(float, tag="4")]
        pub flop_count: f32,
        #[prost(float, tag="5")]
        pub transcendental_count: f32,
        #[prost(float, tag="6")]
        pub bytes_accessed: f32,
        #[prost(float, tag="7")]
        pub optimal_seconds: f32,
        /// The index into the profile counters array for the HloInstruction
        /// corresponding to this HloInstructionInfo.
        #[prost(int64, tag="8")]
        pub profile_index: i64,
    }
    /// Pretty-printer information about an HloComputation.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HloComputationInfo {
        #[prost(string, tag="1")]
        pub name: std::string::String,
        /// The index into the profile counters array for the HloComputation
        /// corresponding to this HloComputationInfo.
        #[prost(int64, tag="2")]
        pub profile_index: i64,
        /// HloInstructionInfos for every HloInstruction in the HloComputation for
        /// corresponding to this HloComputattionInfo.
        #[prost(message, repeated, tag="3")]
        pub instruction_infos: ::std::vec::Vec<HloInstructionInfo>,
    }
}
/// Options for the HLO insert-reduce-precision-operations pass.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HloReducePrecisionOptions {
    #[prost(enumeration="hlo_reduce_precision_options::Location", tag="1")]
    pub location: i32,
    /// Exponent and mantissa bit counts for the reduced precision.
    #[prost(uint32, tag="2")]
    pub exponent_bits: u32,
    #[prost(uint32, tag="3")]
    pub mantissa_bits: u32,
    /// Operations matching these opcodes should be suffixed with reduce-precision
    /// operations.
    #[prost(uint32, repeated, tag="4")]
    pub opcodes_to_suffix: ::std::vec::Vec<u32>,
    /// Operations with names containing these substrings should be suffixed with
    /// reduce-precision operations.
    #[prost(string, repeated, tag="5")]
    pub opname_substrings_to_suffix: ::std::vec::Vec<std::string::String>,
}
pub mod hlo_reduce_precision_options {
    /// Where and when the reduce-precision operations will be added.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Location {
        /// Add reduce-precision operations to the inputs of selected instructions.
        /// This is done before any optimization occurs.
        OpInputs = 0,
        /// Add reduce-precision operations to the outputs of selected instructions.
        /// This is done before any optimization occurs.
        OpOutputs = 1,
        /// After operation-fusion occurs, add reduce-precision operations to the
        /// outputs of any selected instructions that have not been fused into
        /// fusion instructions.
        UnfusedOpOutputs = 2,
        /// After operation-fusion occurs, add reduce-precision operations to the
        /// outputs of any fusion instructions that contain operations matching the
        /// selection criteria.
        FusionInputsByContent = 3,
        /// After operation-fusion occurs, add reduce-precision operations to the
        /// outputs of any fusion instructions that contain operations matching the
        /// selection criteria.
        FusionOutputsByContent = 4,
    }
}
/// Debugging options for XLA. These options may change at any time - there are
/// no guarantees about backward or forward compatibility for these fields.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebugOptions {
    /// Show addresses of HLO ops in graph dump.
    #[prost(bool, tag="2")]
    pub xla_hlo_graph_addresses: bool,
    /// Instrument the computation to collect per-HLO cycle counts.
    #[prost(bool, tag="9")]
    pub xla_hlo_profile: bool,
    /// List of HLO passes to disable/enable. These names must exactly match the
    /// pass names as specified by the HloPassInterface::name() method.
    ///
    /// At least one of xla_disable_hlo_passes and xla_enable_hlo_passes_only must
    /// be empty.
    #[prost(string, repeated, tag="30")]
    pub xla_disable_hlo_passes: ::std::vec::Vec<std::string::String>,
    #[prost(string, repeated, tag="124")]
    pub xla_enable_hlo_passes_only: ::std::vec::Vec<std::string::String>,
    /// Disables all HLO passes.  Notes that some passes are necessary for
    /// correctness and the invariants that must be satisfied by "fully optimized"
    /// HLO are different for different devices and may change over time.  The only
    /// "guarantee", such as it is, is that if you compile XLA and dump the
    /// optimized HLO for some graph, you should be able to run it again on the
    /// same device with the same build of XLA.
    #[prost(bool, tag="104")]
    pub xla_disable_all_hlo_passes: bool,
    /// Numerical optimization level for the XLA compiler backend; the specific
    /// interpretation of this value is left to the backends.
    #[prost(int32, tag="31")]
    pub xla_backend_optimization_level: i32,
    /// Embed the compiler IR as a string in the executable.
    #[prost(bool, tag="33")]
    pub xla_embed_ir_in_executable: bool,
    /// Eliminate implicit broadcasts when lowering user computations to HLO
    /// instructions; use explicit broadcast instead.
    #[prost(bool, tag="35")]
    pub xla_eliminate_hlo_implicit_broadcast: bool,
    /// When generating calls to Eigen in the CPU backend, use multi-threaded Eigen
    /// mode.
    #[prost(bool, tag="60")]
    pub xla_cpu_multi_thread_eigen: bool,
    /// Path to directory with cuda/ptx tools and libraries.
    #[prost(string, tag="61")]
    pub xla_gpu_cuda_data_dir: std::string::String,
    /// Enable flush-to-zero semantics in the GPU backend.
    #[prost(bool, tag="62")]
    pub xla_gpu_ftz: bool,
    /// Disable multi-streaming in the GPU backend.
    #[prost(bool, tag="63")]
    pub xla_gpu_disable_multi_streaming: bool,
    /// If true, in LLVM-based backends, emit !alias.scope metadata in
    /// generated IR.
    #[prost(bool, tag="70")]
    pub xla_llvm_enable_alias_scope_metadata: bool,
    /// If true, in LLVM-based backends, emit !noalias metadata in the
    /// generated IR.
    #[prost(bool, tag="71")]
    pub xla_llvm_enable_noalias_metadata: bool,
    /// If true, in LLVM-based backends, emit !invariant.load metadata in
    /// the generated IR.
    #[prost(bool, tag="72")]
    pub xla_llvm_enable_invariant_load_metadata: bool,
    /// If true, a set of expensive LLVM optimization passes will not be run.
    #[prost(bool, tag="73")]
    pub xla_llvm_disable_expensive_passes: bool,
    /// Options for inserting reduce-precision operations for numerical
    /// experimentation.  This is a repeated field, as we may want to have
    /// multiple passes with different parameters.
    #[prost(message, repeated, tag="80")]
    pub hlo_reduce_precision_options: ::std::vec::Vec<HloReducePrecisionOptions>,
    /// This is used by ClientLibraryTestBase::ComputeAndCompare*. If true, the
    /// computation will run n! times with all permunations of layouts for the
    /// output shape in rank n. For example, with a 3D shape, all permutations of
    /// the set {0, 1, 2} are tried.
    #[prost(bool, tag="90")]
    pub xla_test_all_output_layouts: bool,
    /// This is used by ClientLibraryTestBase::ComputeAndCompare*. If true, the
    /// computation will run for all permunations of layouts of all input
    /// arguments. For example, with 2 input arguments in 2D and 4D shapes, the
    /// computation will run 2! * 4! times.
    #[prost(bool, tag="91")]
    pub xla_test_all_input_layouts: bool,
    /// Assign colors based on sharding information when generating the Graphviz
    /// HLO graph.
    #[prost(bool, tag="92")]
    pub xla_hlo_graph_sharding_color: bool,
    /// If true, the GPU backend is free to use cudnn for HLO batch normalization
    /// ops.
    #[prost(bool, tag="94")]
    pub xla_gpu_use_cudnn_batchnorm: bool,
    /// Generate calls to MKL-DNN in the CPU backend.
    #[prost(bool, tag="97")]
    pub xla_cpu_use_mkl_dnn: bool,
    /// Maximum kernel unroll factor for the GPU backend.
    #[prost(int32, tag="98")]
    pub xla_gpu_max_kernel_unroll_factor: i32,
    /// When true, "unsafe" mathematical optimizations are enabled. These
    /// transformations include but are not limited to:
    ///
    ///  - Reducing the precision of operations (e.g. using an approximate sin
    ///    function, or transforming x/y into x * (1/y)).
    ///  - Assuming that operations never produce or consume NaN or +/- Inf (this
    ///    behavior can be adjusted using xla_cpu_fast_math_allow_{nans|infs}).
    ///  - Assuming that +0 and -0 are indistinguishable.
    #[prost(bool, tag="99")]
    pub xla_cpu_enable_fast_math: bool,
    /// When xla_cpu_enable_fast_math is true then this controls whether we allow
    /// operations to produce NaNs.  Ignored when xla_cpu_enable_fast_math is
    /// false.
    #[prost(bool, tag="120")]
    pub xla_cpu_fast_math_honor_nans: bool,
    /// When xla_cpu_enable_fast_math is true then this controls whether we allow
    /// operations to produce infinites. Ignored when xla_cpu_enable_fast_math is
    /// false.
    #[prost(bool, tag="121")]
    pub xla_cpu_fast_math_honor_infs: bool,
    /// When xla_cpu_enable_fast_math is true then this controls whether we forbid
    /// to use the reciprocal of an argument instead of division. Ignored when
    /// xla_cpu_enable_fast_math is false.
    #[prost(bool, tag="126")]
    pub xla_cpu_fast_math_honor_division: bool,
    /// When xla_cpu_enable_fast_math is true then this controls whether we forbid
    /// to approximate calculations for functions. Ignored when
    /// xla_cpu_enable_fast_math is false.
    #[prost(bool, tag="129")]
    pub xla_cpu_fast_math_honor_functions: bool,
    /// When true we lower the Minimum and Maximum hlos in the GPU backend such
    /// that Min(NotNaN, NaN) = Min(NaN, NotNaN) = NotNaN.  In other words, if flag
    /// this is true we don't propagate NaNs through Min and Max.
    #[prost(bool, tag="100")]
    pub xla_gpu_enable_fast_min_max: bool,
    /// Allows xla to increase the output precision of floating point operations.
    #[prost(bool, tag="122")]
    pub xla_allow_excess_precision: bool,
    /// Crashes the program when any kind of verification fails, instead of just
    /// logging the failures. One example is cross checking of convolution results
    /// among different algorithms.
    #[prost(bool, tag="101")]
    pub xla_gpu_crash_on_verification_failures: bool,
    /// Disable GEMM and Convolution auto-tuning.
    #[prost(bool, tag="123")]
    pub xla_gpu_disable_autotune: bool,
    /// Force the host platform to pretend that there are these many host
    /// "devices".  All these devices are backed by the same threadpool.  Defaults
    /// to 1.
    ///
    /// Setting this to anything other than 1 can increase overhead from context
    /// switching but we let the user override this behavior to help run tests on
    /// the host that run models in parallel across multiple devices.
    #[prost(int32, tag="102")]
    pub xla_force_host_platform_device_count: i32,
    /// If set to true XLA:GPU invokes `ptxas` with -O0 (default is -O3).
    #[prost(bool, tag="103")]
    pub xla_gpu_disable_ptxas_optimizations: bool,
    /// Enable fast math with eigen in the HLO evaluator.
    #[prost(bool, tag="106")]
    pub xla_hlo_evaluator_use_fast_path: bool,
    /// Temporary option to allow support for both the R1 and the scalar index
    /// versions of DynamicSlice and DynamicUpdateSlice. Only used for testing.
    #[prost(bool, tag="107")]
    pub xla_allow_scalar_index_dynamic_ops: bool,
    /// Option to emit a target-specific marker to indicate the start of a training
    /// step. The location of the marker (if any) is determined by the option
    /// value.
    #[prost(enumeration="debug_options::StepMarkerLocation", tag="108")]
    pub xla_step_marker_location: i32,
    //
    // BEGIN flags controlling dumping HLO modules for debugging.
    //
    // When dumping is enabled, HLO modules dumped at the very beginning and end
    // of compilation, and optionally also during the pass pipeline.
    //
    // In general, if you set one of these flags, we will try to infer reasonable
    // defaults for the others.  For example:
    //
    //  * Setting --xla_dump_to=/tmp/foo without specifying a format
    //    with --xla_dump_hlo_as_* will turn on --xla_dump_hlo_as_text.
    //
    //  * Setting --xla_dump_hlo_as_text without specifying --xla_dump_to will
    //    dump to stdout.
    //

    /// Directory to dump into.
    #[prost(string, tag="109")]
    pub xla_dump_to: std::string::String,
    /// If specified, will only dump modules which match this regexp.
    #[prost(string, tag="110")]
    pub xla_dump_hlo_module_re: std::string::String,
    /// If this flag is specified, will also HLO before and after passes that match
    /// this regular expression.  Set to .* to dump before/after all passes.
    #[prost(string, tag="111")]
    pub xla_dump_hlo_pass_re: std::string::String,
    /// Specifies the format that HLO is dumped in.  Multiple of these may be
    /// specified.
    #[prost(bool, tag="112")]
    pub xla_dump_hlo_as_text: bool,
    #[prost(bool, tag="113")]
    pub xla_dump_hlo_as_proto: bool,
    #[prost(bool, tag="114")]
    pub xla_dump_hlo_as_dot: bool,
    #[prost(bool, tag="115")]
    pub xla_dump_hlo_as_url: bool,
    /// Dump HLO graphs as an HTML (DOT -> SVG inlined in HTML)
    #[prost(bool, tag="116")]
    pub xla_dump_hlo_as_html: bool,
    /// If true, every time an HLO module is run, we will dump an HloSnapshot
    /// (essentially, a serialized module plus its inputs) to the --xla_dump_to
    /// directory.
    #[prost(bool, tag="118")]
    pub xla_dump_hlo_snapshots: bool,
    //
    // END flags controlling dumping HLO modules.
    //

    #[prost(bool, tag="125")]
    pub xla_gpu_force_conv_nchw: bool,
    /// Paths to files with ptx code.
    #[prost(string, repeated, tag="127")]
    pub xla_gpu_ptx_file: ::std::vec::Vec<std::string::String>,
    /// Blacklist for cuDNN convolutions.
    #[prost(string, tag="128")]
    pub xla_gpu_algorithm_blacklist_path: std::string::String,
    // Next id: 130

    /// Extra options to pass to the compilation backend (e.g. LLVM); specific
    /// interpretation of these values is left to the backend.
    #[prost(map="string, string", tag="500")]
    pub xla_backend_extra_options: ::std::collections::HashMap<std::string::String, std::string::String>,
}
pub mod debug_options {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum StepMarkerLocation {
        /// Generate a step marker at the program entry. This handles the case where
        /// each step is done by one or multiple program execution(s). Only the first
        /// program will be tagged for generating a step marker at the program entry.
        /// This is the default.
        StepMarkAtEntry = 0,
        /// Generate a step marker at each iteration of the top level while loop,
        /// which is assumed to be a training loop.
        StepMarkAtTopLevelWhileLoop = 1,
        /// Generate a step marker at each iteration of the second level while loops,
        /// which is assumed to be a training or eval loop.
        StepMarkAtSecondLevelWhileLoop = 3,
        /// No step marker generated.
        StepMarkNone = 2,
    }
}
/// These settings control how XLA compiles and/or runs code.  Not all settings
/// will have an effect on every platform.
///
/// When adding new fields, keep in mind that boolean fields default to false.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutionOptions {
    /// This optional field's layout is used as a hint when storing the output of
    /// this computation.  Subsequent transfers of this output array to the client
    /// may be faster when using this layout.
    ///
    /// We use a Shape here to accommodate computations that return a tuple.
    #[prost(message, optional, tag="2")]
    pub shape_with_output_layout: ::std::option::Option<ShapeProto>,
    /// Used to seed random-number generators used in this computation.  If this is
    /// 0, we generate a seed ourselves.
    ///
    /// TODO(b/32083678): Changing the seed unnecessarily forces a recompilation.
    #[prost(uint64, tag="3")]
    pub seed: u64,
    #[prost(message, optional, tag="4")]
    pub debug_options: ::std::option::Option<DebugOptions>,
    /// This optional field specifies a particular set of devices to run the
    /// computation on. The computation will be partitioned across these devices.
    /// If not provided, the default device will be chosen.
    #[prost(message, repeated, tag="5")]
    pub device_handles: ::std::vec::Vec<DeviceHandle>,
    /// Number of replicas of the computation to run. If zero, uses the default
    /// number of replicas for the XLA service.
    #[prost(int32, tag="6")]
    pub num_replicas: i32,
    /// This optional field specifies the device assignment if known at compile
    /// time.
    #[prost(message, optional, tag="7")]
    pub device_assignment: ::std::option::Option<DeviceAssignmentProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDeviceHandlesRequest {
    #[prost(int64, tag="1")]
    pub device_count: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDeviceHandlesResponse {
    #[prost(message, repeated, tag="1")]
    pub device_handles: ::std::vec::Vec<DeviceHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToClientRequest {
    #[prost(message, optional, tag="1")]
    pub data: ::std::option::Option<GlobalDataHandle>,
    /// This optional field directs the service to return the literal in this
    /// layout. A shape is used to hold the layout to accommodate tuples.
    #[prost(message, optional, tag="2")]
    pub shape_with_layout: ::std::option::Option<ShapeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToClientResponse {
    #[prost(message, optional, tag="1")]
    pub literal: ::std::option::Option<LiteralProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToServerRequest {
    #[prost(message, optional, tag="1")]
    pub literal: ::std::option::Option<LiteralProto>,
    #[prost(message, optional, tag="2")]
    pub device_handle: ::std::option::Option<DeviceHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToServerResponse {
    #[prost(message, optional, tag="1")]
    pub data: ::std::option::Option<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToInfeedRequest {
    #[prost(message, optional, tag="1")]
    pub literal: ::std::option::Option<LiteralProto>,
    #[prost(int64, tag="2")]
    pub replica_id: i64,
    #[prost(message, optional, tag="3")]
    pub device_handle: ::std::option::Option<DeviceHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferToInfeedResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferFromOutfeedRequest {
    /// This optional field directs the service to return the literal in this
    /// layout. A shape is used to hold the layout to accommodate tuples.
    #[prost(message, optional, tag="1")]
    pub shape_with_layout: ::std::option::Option<ShapeProto>,
    #[prost(int64, tag="2")]
    pub replica_id: i64,
    #[prost(message, optional, tag="3")]
    pub device_handle: ::std::option::Option<DeviceHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TransferFromOutfeedResponse {
    #[prost(message, optional, tag="1")]
    pub literal: ::std::option::Option<LiteralProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResetDeviceRequest {
    #[prost(message, optional, tag="1")]
    pub device_handle: ::std::option::Option<DeviceHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResetDeviceResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputationGraphStatsRequest {
    #[prost(message, optional, tag="1")]
    pub computation: ::std::option::Option<HloModuleProto>,
    #[prost(message, optional, tag="2")]
    pub debug_options: ::std::option::Option<DebugOptions>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputationStatsResponse {
    #[prost(message, optional, tag="1")]
    pub stats: ::std::option::Option<ComputationStats>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateChannelHandleRequest {
    #[prost(enumeration="channel_handle::ChannelType", tag="1")]
    pub channel_type: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateChannelHandleResponse {
    #[prost(message, optional, tag="1")]
    pub channel: ::std::option::Option<ChannelHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UnregisterRequest {
    #[prost(message, repeated, tag="1")]
    pub data: ::std::vec::Vec<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UnregisterResponse {
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompileRequest {
    /// The graph to be compiled.
    #[prost(message, optional, tag="1")]
    pub computation: ::std::option::Option<HloModuleProto>,
    /// Options that affect how XLA compiles code to service this request.
    #[prost(message, optional, tag="2")]
    pub execution_options: ::std::option::Option<ExecutionOptions>,
    /// The layouts of the input arguments. If not set, the default layout will be
    /// used. Although the real arguments are not needed in compilation, the
    /// layouts of the arguments can affect the compilation.
    #[prost(message, repeated, tag="3")]
    pub input_shape_with_layout: ::std::vec::Vec<ShapeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompileResponse {
    /// The handle to the executable.
    #[prost(message, optional, tag="1")]
    pub handle: ::std::option::Option<ExecutionHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteRequest {
    #[prost(message, optional, tag="1")]
    pub handle: ::std::option::Option<ExecutionHandle>,
    /// The shape and layout of the arguments must be the same as the those of the
    /// executable's parameters.
    #[prost(message, repeated, tag="2")]
    pub arguments: ::std::vec::Vec<GlobalDataHandle>,
}
/// TODO(b/118493728): Remove this and ExecuteGraphParallelRequest and replace
/// the uses with calls to Compile and Execute.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteGraphRequest {
    #[prost(message, optional, tag="1")]
    pub computation: ::std::option::Option<HloModuleProto>,
    #[prost(message, repeated, tag="2")]
    pub arguments: ::std::vec::Vec<GlobalDataHandle>,
    /// Options that affect how XLA compiles and runs code to service this request.
    #[prost(message, optional, tag="3")]
    pub execution_options: ::std::option::Option<ExecutionOptions>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteGraphParallelRequest {
    #[prost(message, repeated, tag="1")]
    pub requests: ::std::vec::Vec<ExecuteGraphRequest>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteResponse {
    #[prost(message, optional, tag="1")]
    pub output: ::std::option::Option<GlobalDataHandle>,
    #[prost(message, optional, tag="2")]
    pub profile: ::std::option::Option<ExecutionProfile>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteParallelResponse {
    #[prost(message, repeated, tag="1")]
    pub responses: ::std::vec::Vec<ExecuteResponse>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WaitForExecutionRequest {
    #[prost(message, optional, tag="1")]
    pub execution: ::std::option::Option<ExecutionHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WaitForExecutionResponse {
    #[prost(message, optional, tag="1")]
    pub output: ::std::option::Option<GlobalDataHandle>,
    #[prost(message, optional, tag="2")]
    pub profile: ::std::option::Option<ExecutionProfile>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputeConstantGraphRequest {
    #[prost(message, optional, tag="1")]
    pub computation: ::std::option::Option<HloModuleProto>,
    #[prost(message, optional, tag="2")]
    pub output_layout: ::std::option::Option<LayoutProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputeConstantResponse {
    /// A LiteralProto is returned directly for this request.
    #[prost(message, optional, tag="1")]
    pub literal: ::std::option::Option<LiteralProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeconstructTupleRequest {
    #[prost(message, optional, tag="2")]
    pub tuple_handle: ::std::option::Option<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeconstructTupleResponse {
    #[prost(message, repeated, tag="1")]
    pub element_handles: ::std::vec::Vec<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LoadDataRequest {
    /// Describes the path of the ColumnIO tablet to load.
    #[prost(string, tag="1")]
    pub columnio_tablet_path: std::string::String,
    /// Describes the field to load within the ColumnIO tablet.
    #[prost(string, tag="2")]
    pub columnio_field: std::string::String,
    /// Individual element shape, excluding rows.
    #[prost(message, optional, tag="3")]
    pub element_shape: ::std::option::Option<ShapeProto>,
    /// Warning: ColumnIO does not support random-access, so use offset with
    /// caution in performance-critical scenarios.
    #[prost(int64, tag="4")]
    pub offset: i64,
    /// Maximum number of elements (with shape element_shape) to load.
    #[prost(int64, tag="5")]
    pub limit: i64,
    /// If more than one item is requested (via limit > 1), then this request
    /// attribute zips together the produced vectors.
    #[prost(bool, tag="6")]
    pub zip: bool,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LoadDataResponse {
    #[prost(message, optional, tag="1")]
    pub data: ::std::option::Option<GlobalDataHandle>,
    #[prost(message, optional, tag="2")]
    pub data_shape: ::std::option::Option<ShapeProto>,
    #[prost(int64, tag="3")]
    pub available_rows: i64,
    #[prost(int64, tag="4")]
    pub rows_loaded: i64,
    #[prost(int64, tag="5")]
    pub nanoseconds: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetShapeRequest {
    #[prost(message, optional, tag="1")]
    pub data: ::std::option::Option<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetShapeResponse {
    #[prost(message, optional, tag="1")]
    pub shape: ::std::option::Option<ShapeProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UnpackRequest {
    #[prost(message, optional, tag="1")]
    pub data: ::std::option::Option<GlobalDataHandle>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UnpackResponse {
    #[prost(message, repeated, tag="1")]
    pub tied_data: ::std::vec::Vec<GlobalDataHandle>,
}
