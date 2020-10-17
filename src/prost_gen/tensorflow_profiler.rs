/// A 'Trace' contains metadata for the individual traces of a system.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Trace {
    /// The devices that this trace has information about. Maps from device_id to
    /// more data about the specific device.
    #[prost(map="uint32, message", tag="1")]
    pub devices: ::std::collections::HashMap<u32, Device>,
    /// All trace events capturing in the profiling period.
    #[prost(message, repeated, tag="4")]
    pub trace_events: ::std::vec::Vec<TraceEvent>,
}
/// A 'device' is a physical entity in the system and is comprised of several
/// resources.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Device {
    /// The name of the device.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The id of this device, unique in a single trace.
    #[prost(uint32, tag="2")]
    pub device_id: u32,
    /// The resources on this device, keyed by resource_id;
    #[prost(map="uint32, message", tag="3")]
    pub resources: ::std::collections::HashMap<u32, Resource>,
}
/// A 'resource' generally is a specific computation component on a device. These
/// can range from threads on CPUs to specific arithmetic units on hardware
/// devices.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Resource {
    /// The name of the resource.
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// The id of the resource. Unique within a device.
    #[prost(uint32, tag="2")]
    pub resource_id: u32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TraceEvent {
    /// The id of the device that this event occurred on. The full dataset should
    /// have this device present in the Trace object.
    #[prost(uint32, tag="1")]
    pub device_id: u32,
    /// The id of the resource that this event occurred on. The full dataset should
    /// have this resource present in the Device object of the Trace object. A
    /// resource_id is unique on a specific device, but not necessarily within the
    /// trace.
    #[prost(uint32, tag="2")]
    pub resource_id: u32,
    /// The name of this trace event.
    #[prost(string, tag="3")]
    pub name: std::string::String,
    /// The timestamp that this event occurred at (in picos since tracing started).
    #[prost(uint64, tag="9")]
    pub timestamp_ps: u64,
    /// The duration of the event in picoseconds if applicable.
    /// Events without duration are called instant events.
    #[prost(uint64, tag="10")]
    pub duration_ps: u64,
    /// Extra arguments that will be displayed in trace view.
    #[prost(map="string, string", tag="11")]
    pub args: ::std::collections::HashMap<std::string::String, std::string::String>,
}
