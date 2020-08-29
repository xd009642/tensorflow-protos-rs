// This file is generated by rust-protobuf 2.17.0. Do not edit
// @generated

// https://github.com/rust-lang/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![allow(unused_attributes)]
#![rustfmt::skip]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unused_imports)]
#![allow(unused_results)]
//! Generated file from `tensorflow/core/protobuf/trace_events.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct Trace {
    // message fields
    pub devices: ::std::collections::HashMap<u32, Device>,
    pub trace_events: ::protobuf::RepeatedField<TraceEvent>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a Trace {
    fn default() -> &'a Trace {
        <Trace as ::protobuf::Message>::default_instance()
    }
}

impl Trace {
    pub fn new() -> Trace {
        ::std::default::Default::default()
    }

    // repeated .tensorflow.profiler.Trace.DevicesEntry devices = 1;


    pub fn get_devices(&self) -> &::std::collections::HashMap<u32, Device> {
        &self.devices
    }
    pub fn clear_devices(&mut self) {
        self.devices.clear();
    }

    // Param is passed by value, moved
    pub fn set_devices(&mut self, v: ::std::collections::HashMap<u32, Device>) {
        self.devices = v;
    }

    // Mutable pointer to the field.
    pub fn mut_devices(&mut self) -> &mut ::std::collections::HashMap<u32, Device> {
        &mut self.devices
    }

    // Take field
    pub fn take_devices(&mut self) -> ::std::collections::HashMap<u32, Device> {
        ::std::mem::replace(&mut self.devices, ::std::collections::HashMap::new())
    }

    // repeated .tensorflow.profiler.TraceEvent trace_events = 4;


    pub fn get_trace_events(&self) -> &[TraceEvent] {
        &self.trace_events
    }
    pub fn clear_trace_events(&mut self) {
        self.trace_events.clear();
    }

    // Param is passed by value, moved
    pub fn set_trace_events(&mut self, v: ::protobuf::RepeatedField<TraceEvent>) {
        self.trace_events = v;
    }

    // Mutable pointer to the field.
    pub fn mut_trace_events(&mut self) -> &mut ::protobuf::RepeatedField<TraceEvent> {
        &mut self.trace_events
    }

    // Take field
    pub fn take_trace_events(&mut self) -> ::protobuf::RepeatedField<TraceEvent> {
        ::std::mem::replace(&mut self.trace_events, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for Trace {
    fn is_initialized(&self) -> bool {
        for v in &self.trace_events {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_map_into::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Device>>(wire_type, is, &mut self.devices)?;
                },
                4 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.trace_events)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        my_size += ::protobuf::rt::compute_map_size::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Device>>(1, &self.devices);
        for value in &self.trace_events {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        ::protobuf::rt::write_map_with_cached_sizes::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Device>>(1, &self.devices, os)?;
        for v in &self.trace_events {
            os.write_tag(4, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> Trace {
        Trace::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_map_accessor::<_, ::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Device>>(
                "devices",
                |m: &Trace| { &m.devices },
                |m: &mut Trace| { &mut m.devices },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TraceEvent>>(
                "trace_events",
                |m: &Trace| { &m.trace_events },
                |m: &mut Trace| { &mut m.trace_events },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<Trace>(
                "Trace",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static Trace {
        static instance: ::protobuf::rt::LazyV2<Trace> = ::protobuf::rt::LazyV2::INIT;
        instance.get(Trace::new)
    }
}

impl ::protobuf::Clear for Trace {
    fn clear(&mut self) {
        self.devices.clear();
        self.trace_events.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for Trace {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for Trace {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct Device {
    // message fields
    pub name: ::std::string::String,
    pub device_id: u32,
    pub resources: ::std::collections::HashMap<u32, Resource>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a Device {
    fn default() -> &'a Device {
        <Device as ::protobuf::Message>::default_instance()
    }
}

impl Device {
    pub fn new() -> Device {
        ::std::default::Default::default()
    }

    // string name = 1;


    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        &mut self.name
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.name, ::std::string::String::new())
    }

    // uint32 device_id = 2;


    pub fn get_device_id(&self) -> u32 {
        self.device_id
    }
    pub fn clear_device_id(&mut self) {
        self.device_id = 0;
    }

    // Param is passed by value, moved
    pub fn set_device_id(&mut self, v: u32) {
        self.device_id = v;
    }

    // repeated .tensorflow.profiler.Device.ResourcesEntry resources = 3;


    pub fn get_resources(&self) -> &::std::collections::HashMap<u32, Resource> {
        &self.resources
    }
    pub fn clear_resources(&mut self) {
        self.resources.clear();
    }

    // Param is passed by value, moved
    pub fn set_resources(&mut self, v: ::std::collections::HashMap<u32, Resource>) {
        self.resources = v;
    }

    // Mutable pointer to the field.
    pub fn mut_resources(&mut self) -> &mut ::std::collections::HashMap<u32, Resource> {
        &mut self.resources
    }

    // Take field
    pub fn take_resources(&mut self) -> ::std::collections::HashMap<u32, Resource> {
        ::std::mem::replace(&mut self.resources, ::std::collections::HashMap::new())
    }
}

impl ::protobuf::Message for Device {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.name)?;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.device_id = tmp;
                },
                3 => {
                    ::protobuf::rt::read_map_into::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Resource>>(wire_type, is, &mut self.resources)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if !self.name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.name);
        }
        if self.device_id != 0 {
            my_size += ::protobuf::rt::value_size(2, self.device_id, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::compute_map_size::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Resource>>(3, &self.resources);
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.name.is_empty() {
            os.write_string(1, &self.name)?;
        }
        if self.device_id != 0 {
            os.write_uint32(2, self.device_id)?;
        }
        ::protobuf::rt::write_map_with_cached_sizes::<::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Resource>>(3, &self.resources, os)?;
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> Device {
        Device::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "name",
                |m: &Device| { &m.name },
                |m: &mut Device| { &mut m.name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "device_id",
                |m: &Device| { &m.device_id },
                |m: &mut Device| { &mut m.device_id },
            ));
            fields.push(::protobuf::reflect::accessor::make_map_accessor::<_, ::protobuf::types::ProtobufTypeUint32, ::protobuf::types::ProtobufTypeMessage<Resource>>(
                "resources",
                |m: &Device| { &m.resources },
                |m: &mut Device| { &mut m.resources },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<Device>(
                "Device",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static Device {
        static instance: ::protobuf::rt::LazyV2<Device> = ::protobuf::rt::LazyV2::INIT;
        instance.get(Device::new)
    }
}

impl ::protobuf::Clear for Device {
    fn clear(&mut self) {
        self.name.clear();
        self.device_id = 0;
        self.resources.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for Device {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for Device {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct Resource {
    // message fields
    pub name: ::std::string::String,
    pub resource_id: u32,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a Resource {
    fn default() -> &'a Resource {
        <Resource as ::protobuf::Message>::default_instance()
    }
}

impl Resource {
    pub fn new() -> Resource {
        ::std::default::Default::default()
    }

    // string name = 1;


    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        &mut self.name
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.name, ::std::string::String::new())
    }

    // uint32 resource_id = 2;


    pub fn get_resource_id(&self) -> u32 {
        self.resource_id
    }
    pub fn clear_resource_id(&mut self) {
        self.resource_id = 0;
    }

    // Param is passed by value, moved
    pub fn set_resource_id(&mut self, v: u32) {
        self.resource_id = v;
    }
}

impl ::protobuf::Message for Resource {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.name)?;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.resource_id = tmp;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if !self.name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.name);
        }
        if self.resource_id != 0 {
            my_size += ::protobuf::rt::value_size(2, self.resource_id, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.name.is_empty() {
            os.write_string(1, &self.name)?;
        }
        if self.resource_id != 0 {
            os.write_uint32(2, self.resource_id)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> Resource {
        Resource::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "name",
                |m: &Resource| { &m.name },
                |m: &mut Resource| { &mut m.name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "resource_id",
                |m: &Resource| { &m.resource_id },
                |m: &mut Resource| { &mut m.resource_id },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<Resource>(
                "Resource",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static Resource {
        static instance: ::protobuf::rt::LazyV2<Resource> = ::protobuf::rt::LazyV2::INIT;
        instance.get(Resource::new)
    }
}

impl ::protobuf::Clear for Resource {
    fn clear(&mut self) {
        self.name.clear();
        self.resource_id = 0;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for Resource {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for Resource {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TraceEvent {
    // message fields
    pub device_id: u32,
    pub resource_id: u32,
    pub name: ::std::string::String,
    pub timestamp_ps: u64,
    pub duration_ps: u64,
    pub args: ::std::collections::HashMap<::std::string::String, ::std::string::String>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a TraceEvent {
    fn default() -> &'a TraceEvent {
        <TraceEvent as ::protobuf::Message>::default_instance()
    }
}

impl TraceEvent {
    pub fn new() -> TraceEvent {
        ::std::default::Default::default()
    }

    // uint32 device_id = 1;


    pub fn get_device_id(&self) -> u32 {
        self.device_id
    }
    pub fn clear_device_id(&mut self) {
        self.device_id = 0;
    }

    // Param is passed by value, moved
    pub fn set_device_id(&mut self, v: u32) {
        self.device_id = v;
    }

    // uint32 resource_id = 2;


    pub fn get_resource_id(&self) -> u32 {
        self.resource_id
    }
    pub fn clear_resource_id(&mut self) {
        self.resource_id = 0;
    }

    // Param is passed by value, moved
    pub fn set_resource_id(&mut self, v: u32) {
        self.resource_id = v;
    }

    // string name = 3;


    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        &mut self.name
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.name, ::std::string::String::new())
    }

    // uint64 timestamp_ps = 9;


    pub fn get_timestamp_ps(&self) -> u64 {
        self.timestamp_ps
    }
    pub fn clear_timestamp_ps(&mut self) {
        self.timestamp_ps = 0;
    }

    // Param is passed by value, moved
    pub fn set_timestamp_ps(&mut self, v: u64) {
        self.timestamp_ps = v;
    }

    // uint64 duration_ps = 10;


    pub fn get_duration_ps(&self) -> u64 {
        self.duration_ps
    }
    pub fn clear_duration_ps(&mut self) {
        self.duration_ps = 0;
    }

    // Param is passed by value, moved
    pub fn set_duration_ps(&mut self, v: u64) {
        self.duration_ps = v;
    }

    // repeated .tensorflow.profiler.TraceEvent.ArgsEntry args = 11;


    pub fn get_args(&self) -> &::std::collections::HashMap<::std::string::String, ::std::string::String> {
        &self.args
    }
    pub fn clear_args(&mut self) {
        self.args.clear();
    }

    // Param is passed by value, moved
    pub fn set_args(&mut self, v: ::std::collections::HashMap<::std::string::String, ::std::string::String>) {
        self.args = v;
    }

    // Mutable pointer to the field.
    pub fn mut_args(&mut self) -> &mut ::std::collections::HashMap<::std::string::String, ::std::string::String> {
        &mut self.args
    }

    // Take field
    pub fn take_args(&mut self) -> ::std::collections::HashMap<::std::string::String, ::std::string::String> {
        ::std::mem::replace(&mut self.args, ::std::collections::HashMap::new())
    }
}

impl ::protobuf::Message for TraceEvent {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.device_id = tmp;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.resource_id = tmp;
                },
                3 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.name)?;
                },
                9 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint64()?;
                    self.timestamp_ps = tmp;
                },
                10 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint64()?;
                    self.duration_ps = tmp;
                },
                11 => {
                    ::protobuf::rt::read_map_into::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeString>(wire_type, is, &mut self.args)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if self.device_id != 0 {
            my_size += ::protobuf::rt::value_size(1, self.device_id, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.resource_id != 0 {
            my_size += ::protobuf::rt::value_size(2, self.resource_id, ::protobuf::wire_format::WireTypeVarint);
        }
        if !self.name.is_empty() {
            my_size += ::protobuf::rt::string_size(3, &self.name);
        }
        if self.timestamp_ps != 0 {
            my_size += ::protobuf::rt::value_size(9, self.timestamp_ps, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.duration_ps != 0 {
            my_size += ::protobuf::rt::value_size(10, self.duration_ps, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::compute_map_size::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeString>(11, &self.args);
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.device_id != 0 {
            os.write_uint32(1, self.device_id)?;
        }
        if self.resource_id != 0 {
            os.write_uint32(2, self.resource_id)?;
        }
        if !self.name.is_empty() {
            os.write_string(3, &self.name)?;
        }
        if self.timestamp_ps != 0 {
            os.write_uint64(9, self.timestamp_ps)?;
        }
        if self.duration_ps != 0 {
            os.write_uint64(10, self.duration_ps)?;
        }
        ::protobuf::rt::write_map_with_cached_sizes::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeString>(11, &self.args, os)?;
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> TraceEvent {
        TraceEvent::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "device_id",
                |m: &TraceEvent| { &m.device_id },
                |m: &mut TraceEvent| { &mut m.device_id },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "resource_id",
                |m: &TraceEvent| { &m.resource_id },
                |m: &mut TraceEvent| { &mut m.resource_id },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "name",
                |m: &TraceEvent| { &m.name },
                |m: &mut TraceEvent| { &mut m.name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint64>(
                "timestamp_ps",
                |m: &TraceEvent| { &m.timestamp_ps },
                |m: &mut TraceEvent| { &mut m.timestamp_ps },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint64>(
                "duration_ps",
                |m: &TraceEvent| { &m.duration_ps },
                |m: &mut TraceEvent| { &mut m.duration_ps },
            ));
            fields.push(::protobuf::reflect::accessor::make_map_accessor::<_, ::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeString>(
                "args",
                |m: &TraceEvent| { &m.args },
                |m: &mut TraceEvent| { &mut m.args },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<TraceEvent>(
                "TraceEvent",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static TraceEvent {
        static instance: ::protobuf::rt::LazyV2<TraceEvent> = ::protobuf::rt::LazyV2::INIT;
        instance.get(TraceEvent::new)
    }
}

impl ::protobuf::Clear for TraceEvent {
    fn clear(&mut self) {
        self.device_id = 0;
        self.resource_id = 0;
        self.name.clear();
        self.timestamp_ps = 0;
        self.duration_ps = 0;
        self.args.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TraceEvent {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TraceEvent {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n+tensorflow/core/protobuf/trace_events.proto\x12\x13tensorflow.profile\
    r\"\xe7\x01\n\x05Trace\x12A\n\x07devices\x18\x01\x20\x03(\x0b2'.tensorfl\
    ow.profiler.Trace.DevicesEntryR\x07devices\x12B\n\x0ctrace_events\x18\
    \x04\x20\x03(\x0b2\x1f.tensorflow.profiler.TraceEventR\x0btraceEvents\
    \x1aW\n\x0cDevicesEntry\x12\x10\n\x03key\x18\x01\x20\x01(\rR\x03key\x121\
    \n\x05value\x18\x02\x20\x01(\x0b2\x1b.tensorflow.profiler.DeviceR\x05val\
    ue:\x028\x01\"\xe0\x01\n\x06Device\x12\x12\n\x04name\x18\x01\x20\x01(\tR\
    \x04name\x12\x1b\n\tdevice_id\x18\x02\x20\x01(\rR\x08deviceId\x12H\n\tre\
    sources\x18\x03\x20\x03(\x0b2*.tensorflow.profiler.Device.ResourcesEntry\
    R\tresources\x1a[\n\x0eResourcesEntry\x12\x10\n\x03key\x18\x01\x20\x01(\
    \rR\x03key\x123\n\x05value\x18\x02\x20\x01(\x0b2\x1d.tensorflow.profiler\
    .ResourceR\x05value:\x028\x01\"?\n\x08Resource\x12\x12\n\x04name\x18\x01\
    \x20\x01(\tR\x04name\x12\x1f\n\x0bresource_id\x18\x02\x20\x01(\rR\nresou\
    rceId\"\x9a\x02\n\nTraceEvent\x12\x1b\n\tdevice_id\x18\x01\x20\x01(\rR\
    \x08deviceId\x12\x1f\n\x0bresource_id\x18\x02\x20\x01(\rR\nresourceId\
    \x12\x12\n\x04name\x18\x03\x20\x01(\tR\x04name\x12!\n\x0ctimestamp_ps\
    \x18\t\x20\x01(\x04R\x0btimestampPs\x12\x1f\n\x0bduration_ps\x18\n\x20\
    \x01(\x04R\ndurationPs\x12=\n\x04args\x18\x0b\x20\x03(\x0b2).tensorflow.\
    profiler.TraceEvent.ArgsEntryR\x04args\x1a7\n\tArgsEntry\x12\x10\n\x03ke\
    y\x18\x01\x20\x01(\tR\x03key\x12\x14\n\x05value\x18\x02\x20\x01(\tR\x05v\
    alue:\x028\x01B2\n\x18org.tensorflow.frameworkB\x11TraceEventsProtosP\
    \x01\xf8\x01\x01b\x06proto3\
";

static file_descriptor_proto_lazy: ::protobuf::rt::LazyV2<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::rt::LazyV2::INIT;

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    file_descriptor_proto_lazy.get(|| {
        parse_descriptor_proto()
    })
}
