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
//! Generated file from `tensorflow/core/protobuf/tpu/topology.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct TopologyProto {
    // message fields
    pub mesh_shape: ::std::vec::Vec<i32>,
    pub num_tasks: i32,
    pub num_tpu_devices_per_task: i32,
    pub device_coordinates: ::std::vec::Vec<i32>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a TopologyProto {
    fn default() -> &'a TopologyProto {
        <TopologyProto as ::protobuf::Message>::default_instance()
    }
}

impl TopologyProto {
    pub fn new() -> TopologyProto {
        ::std::default::Default::default()
    }

    // repeated int32 mesh_shape = 1;


    pub fn get_mesh_shape(&self) -> &[i32] {
        &self.mesh_shape
    }
    pub fn clear_mesh_shape(&mut self) {
        self.mesh_shape.clear();
    }

    // Param is passed by value, moved
    pub fn set_mesh_shape(&mut self, v: ::std::vec::Vec<i32>) {
        self.mesh_shape = v;
    }

    // Mutable pointer to the field.
    pub fn mut_mesh_shape(&mut self) -> &mut ::std::vec::Vec<i32> {
        &mut self.mesh_shape
    }

    // Take field
    pub fn take_mesh_shape(&mut self) -> ::std::vec::Vec<i32> {
        ::std::mem::replace(&mut self.mesh_shape, ::std::vec::Vec::new())
    }

    // int32 num_tasks = 2;


    pub fn get_num_tasks(&self) -> i32 {
        self.num_tasks
    }
    pub fn clear_num_tasks(&mut self) {
        self.num_tasks = 0;
    }

    // Param is passed by value, moved
    pub fn set_num_tasks(&mut self, v: i32) {
        self.num_tasks = v;
    }

    // int32 num_tpu_devices_per_task = 3;


    pub fn get_num_tpu_devices_per_task(&self) -> i32 {
        self.num_tpu_devices_per_task
    }
    pub fn clear_num_tpu_devices_per_task(&mut self) {
        self.num_tpu_devices_per_task = 0;
    }

    // Param is passed by value, moved
    pub fn set_num_tpu_devices_per_task(&mut self, v: i32) {
        self.num_tpu_devices_per_task = v;
    }

    // repeated int32 device_coordinates = 4;


    pub fn get_device_coordinates(&self) -> &[i32] {
        &self.device_coordinates
    }
    pub fn clear_device_coordinates(&mut self) {
        self.device_coordinates.clear();
    }

    // Param is passed by value, moved
    pub fn set_device_coordinates(&mut self, v: ::std::vec::Vec<i32>) {
        self.device_coordinates = v;
    }

    // Mutable pointer to the field.
    pub fn mut_device_coordinates(&mut self) -> &mut ::std::vec::Vec<i32> {
        &mut self.device_coordinates
    }

    // Take field
    pub fn take_device_coordinates(&mut self) -> ::std::vec::Vec<i32> {
        ::std::mem::replace(&mut self.device_coordinates, ::std::vec::Vec::new())
    }
}

impl ::protobuf::Message for TopologyProto {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_int32_into(wire_type, is, &mut self.mesh_shape)?;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.num_tasks = tmp;
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.num_tpu_devices_per_task = tmp;
                },
                4 => {
                    ::protobuf::rt::read_repeated_int32_into(wire_type, is, &mut self.device_coordinates)?;
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
        for value in &self.mesh_shape {
            my_size += ::protobuf::rt::value_size(1, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        if self.num_tasks != 0 {
            my_size += ::protobuf::rt::value_size(2, self.num_tasks, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.num_tpu_devices_per_task != 0 {
            my_size += ::protobuf::rt::value_size(3, self.num_tpu_devices_per_task, ::protobuf::wire_format::WireTypeVarint);
        }
        for value in &self.device_coordinates {
            my_size += ::protobuf::rt::value_size(4, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.mesh_shape {
            os.write_int32(1, *v)?;
        };
        if self.num_tasks != 0 {
            os.write_int32(2, self.num_tasks)?;
        }
        if self.num_tpu_devices_per_task != 0 {
            os.write_int32(3, self.num_tpu_devices_per_task)?;
        }
        for v in &self.device_coordinates {
            os.write_int32(4, *v)?;
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

    fn new() -> TopologyProto {
        TopologyProto::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "mesh_shape",
                |m: &TopologyProto| { &m.mesh_shape },
                |m: &mut TopologyProto| { &mut m.mesh_shape },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "num_tasks",
                |m: &TopologyProto| { &m.num_tasks },
                |m: &mut TopologyProto| { &mut m.num_tasks },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "num_tpu_devices_per_task",
                |m: &TopologyProto| { &m.num_tpu_devices_per_task },
                |m: &mut TopologyProto| { &mut m.num_tpu_devices_per_task },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "device_coordinates",
                |m: &TopologyProto| { &m.device_coordinates },
                |m: &mut TopologyProto| { &mut m.device_coordinates },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<TopologyProto>(
                "TopologyProto",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static TopologyProto {
        static instance: ::protobuf::rt::LazyV2<TopologyProto> = ::protobuf::rt::LazyV2::INIT;
        instance.get(TopologyProto::new)
    }
}

impl ::protobuf::Clear for TopologyProto {
    fn clear(&mut self) {
        self.mesh_shape.clear();
        self.num_tasks = 0;
        self.num_tpu_devices_per_task = 0;
        self.device_coordinates.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TopologyProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TopologyProto {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n+tensorflow/core/protobuf/tpu/topology.proto\x12\x0etensorflow.tpu\"\
    \xb2\x01\n\rTopologyProto\x12\x1d\n\nmesh_shape\x18\x01\x20\x03(\x05R\tm\
    eshShape\x12\x1b\n\tnum_tasks\x18\x02\x20\x01(\x05R\x08numTasks\x126\n\
    \x18num_tpu_devices_per_task\x18\x03\x20\x01(\x05R\x14numTpuDevicesPerTa\
    sk\x12-\n\x12device_coordinates\x18\x04\x20\x03(\x05R\x11deviceCoordinat\
    esB\x03\xf8\x01\x01b\x06proto3\
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
