/// Node describes a node in a tree.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Node {
    #[prost(message, optional, tag="777")]
    pub metadata: ::std::option::Option<NodeMetadata>,
    #[prost(oneof="node::Node", tags="1, 2, 3, 4")]
    pub node: ::std::option::Option<node::Node>,
}
pub mod node {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Node {
        #[prost(message, tag="1")]
        Leaf(super::Leaf),
        #[prost(message, tag="2")]
        BucketizedSplit(super::BucketizedSplit),
        #[prost(message, tag="3")]
        CategoricalSplit(super::CategoricalSplit),
        #[prost(message, tag="4")]
        DenseSplit(super::DenseSplit),
    }
}
/// NodeMetadata encodes metadata associated with each node in a tree.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeMetadata {
    /// The gain associated with this node.
    #[prost(float, tag="1")]
    pub gain: f32,
    /// The original leaf node before this node was split.
    #[prost(message, optional, tag="2")]
    pub original_leaf: ::std::option::Option<Leaf>,
}
/// Leaves can either hold dense or sparse information.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Leaf {
    #[prost(float, tag="3")]
    pub scalar: f32,
    #[prost(oneof="leaf::Leaf", tags="1, 2")]
    pub leaf: ::std::option::Option<leaf::Leaf>,
}
pub mod leaf {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Leaf {
        /// See third_party/tensorflow/contrib/decision_trees/
        /// proto/generic_tree_model.proto
        /// for a description of how vector and sparse_vector might be used.
        #[prost(message, tag="1")]
        Vector(super::Vector),
        #[prost(message, tag="2")]
        SparseVector(super::SparseVector),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Vector {
    #[prost(float, repeated, tag="1")]
    pub value: ::std::vec::Vec<f32>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SparseVector {
    #[prost(int32, repeated, tag="1")]
    pub index: ::std::vec::Vec<i32>,
    #[prost(float, repeated, tag="2")]
    pub value: ::std::vec::Vec<f32>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BucketizedSplit {
    /// Float feature column and split threshold describing
    /// the rule feature <= threshold.
    #[prost(int32, tag="1")]
    pub feature_id: i32,
    #[prost(int32, tag="2")]
    pub threshold: i32,
    /// If feature column is multivalent, this holds the index of the dimension
    /// for the split. Defaults to 0.
    #[prost(int32, tag="5")]
    pub dimension_id: i32,
    /// default direction for missing values.
    #[prost(enumeration="DefaultDirection", tag="6")]
    pub default_direction: i32,
    /// Node children indexing into a contiguous
    /// vector of nodes starting from the root.
    #[prost(int32, tag="3")]
    pub left_id: i32,
    #[prost(int32, tag="4")]
    pub right_id: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CategoricalSplit {
    /// Categorical feature column and split describing the rule feature value ==
    /// value.
    #[prost(int32, tag="1")]
    pub feature_id: i32,
    #[prost(int32, tag="2")]
    pub value: i32,
    /// If feature column is multivalent, this holds the index of the dimension
    /// for the split. Defaults to 0.
    #[prost(int32, tag="5")]
    pub dimension_id: i32,
    /// Node children indexing into a contiguous
    /// vector of nodes starting from the root.
    #[prost(int32, tag="3")]
    pub left_id: i32,
    #[prost(int32, tag="4")]
    pub right_id: i32,
}
/// TODO(nponomareva): move out of boosted_trees and rename to trees.proto
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DenseSplit {
    /// Float feature column and split threshold describing
    /// the rule feature <= threshold.
    #[prost(int32, tag="1")]
    pub feature_id: i32,
    #[prost(float, tag="2")]
    pub threshold: f32,
    /// Node children indexing into a contiguous
    /// vector of nodes starting from the root.
    #[prost(int32, tag="3")]
    pub left_id: i32,
    #[prost(int32, tag="4")]
    pub right_id: i32,
}
/// Tree describes a list of connected nodes.
/// Node 0 must be the root and can carry any payload including a leaf
/// in the case of representing the bias.
/// Note that each node id is implicitly its index in the list of nodes.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Tree {
    #[prost(message, repeated, tag="1")]
    pub nodes: ::std::vec::Vec<Node>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TreeMetadata {
    /// Number of layers grown for this tree.
    #[prost(int32, tag="2")]
    pub num_layers_grown: i32,
    /// Whether the tree is finalized in that no more layers can be grown.
    #[prost(bool, tag="3")]
    pub is_finalized: bool,
    /// If tree was finalized and post pruning happened, it is possible that cache
    /// still refers to some nodes that were deleted or that the node ids changed
    /// (e.g. node id 5 became node id 2 due to pruning of the other branch).
    /// The mapping below allows us to understand where the old ids now map to and
    /// how the values should be adjusted due to post-pruning.
    /// The size of the list should be equal to the number of nodes in the tree
    /// before post-pruning happened.
    /// If the node was pruned, it will have new_node_id equal to the id of a node
    /// that this node was collapsed into. For a node that didn't get pruned, it is
    /// possible that its id still changed, so new_node_id will have the
    /// corresponding id in the pruned tree.
    /// If post-pruning didn't happen, or it did and it had no effect (e.g. no
    /// nodes got pruned), this list will be empty.
    #[prost(message, repeated, tag="4")]
    pub post_pruned_nodes_meta: ::std::vec::Vec<tree_metadata::PostPruneNodeUpdate>,
}
pub mod tree_metadata {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct PostPruneNodeUpdate {
        #[prost(int32, tag="1")]
        pub new_node_id: i32,
        #[prost(float, repeated, tag="2")]
        pub logit_change: ::std::vec::Vec<f32>,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrowingMetadata {
    /// Number of trees that we have attempted to build. After pruning, these
    /// trees might have been removed.
    #[prost(int64, tag="1")]
    pub num_trees_attempted: i64,
    /// Number of layers that we have attempted to build. After pruning, these
    /// layers might have been removed.
    #[prost(int64, tag="2")]
    pub num_layers_attempted: i64,
    /// The start (inclusive) and end (exclusive) ids of the nodes in the latest
    /// layer of the latest tree.
    #[prost(int32, tag="3")]
    pub last_layer_node_start: i32,
    #[prost(int32, tag="4")]
    pub last_layer_node_end: i32,
}
/// TreeEnsemble describes an ensemble of decision trees.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TreeEnsemble {
    #[prost(message, repeated, tag="1")]
    pub trees: ::std::vec::Vec<Tree>,
    #[prost(float, repeated, tag="2")]
    pub tree_weights: ::std::vec::Vec<f32>,
    #[prost(message, repeated, tag="3")]
    pub tree_metadata: ::std::vec::Vec<TreeMetadata>,
    /// Metadata that is used during the training.
    #[prost(message, optional, tag="4")]
    pub growing_metadata: ::std::option::Option<GrowingMetadata>,
}
/// DebugOutput contains outputs useful for debugging/model interpretation, at
/// the individual example-level. Debug outputs that are available to the user
/// are: 1) Directional feature contributions (DFCs) 2) Node IDs for ensemble
/// prediction path 3) Leaf node IDs.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebugOutput {
    /// Return the logits and associated feature splits across prediction paths for
    /// each tree, for every example, at predict time. We will use these values to
    /// compute DFCs in Python, by subtracting each child prediction from its
    /// parent prediction and associating this change with its respective feature
    /// id.
    #[prost(int32, repeated, tag="1")]
    pub feature_ids: ::std::vec::Vec<i32>,
    #[prost(float, repeated, tag="2")]
    pub logits_path: ::std::vec::Vec<f32>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum SplitTypeWithDefault {
    InequalityDefaultLeft = 0,
    InequalityDefaultRight = 1,
    EqualityDefaultRight = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum DefaultDirection {
    /// Left is the default direction.
    DefaultLeft = 0,
    DefaultRight = 1,
}
