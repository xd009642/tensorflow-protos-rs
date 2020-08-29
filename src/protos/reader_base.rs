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
//! Generated file from `tensorflow/core/framework/reader_base.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct ReaderBaseState {
    // message fields
    pub work_started: i64,
    pub work_finished: i64,
    pub num_records_produced: i64,
    pub current_work: ::std::vec::Vec<u8>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a ReaderBaseState {
    fn default() -> &'a ReaderBaseState {
        <ReaderBaseState as ::protobuf::Message>::default_instance()
    }
}

impl ReaderBaseState {
    pub fn new() -> ReaderBaseState {
        ::std::default::Default::default()
    }

    // int64 work_started = 1;


    pub fn get_work_started(&self) -> i64 {
        self.work_started
    }
    pub fn clear_work_started(&mut self) {
        self.work_started = 0;
    }

    // Param is passed by value, moved
    pub fn set_work_started(&mut self, v: i64) {
        self.work_started = v;
    }

    // int64 work_finished = 2;


    pub fn get_work_finished(&self) -> i64 {
        self.work_finished
    }
    pub fn clear_work_finished(&mut self) {
        self.work_finished = 0;
    }

    // Param is passed by value, moved
    pub fn set_work_finished(&mut self, v: i64) {
        self.work_finished = v;
    }

    // int64 num_records_produced = 3;


    pub fn get_num_records_produced(&self) -> i64 {
        self.num_records_produced
    }
    pub fn clear_num_records_produced(&mut self) {
        self.num_records_produced = 0;
    }

    // Param is passed by value, moved
    pub fn set_num_records_produced(&mut self, v: i64) {
        self.num_records_produced = v;
    }

    // bytes current_work = 4;


    pub fn get_current_work(&self) -> &[u8] {
        &self.current_work
    }
    pub fn clear_current_work(&mut self) {
        self.current_work.clear();
    }

    // Param is passed by value, moved
    pub fn set_current_work(&mut self, v: ::std::vec::Vec<u8>) {
        self.current_work = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_current_work(&mut self) -> &mut ::std::vec::Vec<u8> {
        &mut self.current_work
    }

    // Take field
    pub fn take_current_work(&mut self) -> ::std::vec::Vec<u8> {
        ::std::mem::replace(&mut self.current_work, ::std::vec::Vec::new())
    }
}

impl ::protobuf::Message for ReaderBaseState {
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
                    let tmp = is.read_int64()?;
                    self.work_started = tmp;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.work_finished = tmp;
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.num_records_produced = tmp;
                },
                4 => {
                    ::protobuf::rt::read_singular_proto3_bytes_into(wire_type, is, &mut self.current_work)?;
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
        if self.work_started != 0 {
            my_size += ::protobuf::rt::value_size(1, self.work_started, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.work_finished != 0 {
            my_size += ::protobuf::rt::value_size(2, self.work_finished, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.num_records_produced != 0 {
            my_size += ::protobuf::rt::value_size(3, self.num_records_produced, ::protobuf::wire_format::WireTypeVarint);
        }
        if !self.current_work.is_empty() {
            my_size += ::protobuf::rt::bytes_size(4, &self.current_work);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.work_started != 0 {
            os.write_int64(1, self.work_started)?;
        }
        if self.work_finished != 0 {
            os.write_int64(2, self.work_finished)?;
        }
        if self.num_records_produced != 0 {
            os.write_int64(3, self.num_records_produced)?;
        }
        if !self.current_work.is_empty() {
            os.write_bytes(4, &self.current_work)?;
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

    fn new() -> ReaderBaseState {
        ReaderBaseState::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "work_started",
                |m: &ReaderBaseState| { &m.work_started },
                |m: &mut ReaderBaseState| { &mut m.work_started },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "work_finished",
                |m: &ReaderBaseState| { &m.work_finished },
                |m: &mut ReaderBaseState| { &mut m.work_finished },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "num_records_produced",
                |m: &ReaderBaseState| { &m.num_records_produced },
                |m: &mut ReaderBaseState| { &mut m.num_records_produced },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBytes>(
                "current_work",
                |m: &ReaderBaseState| { &m.current_work },
                |m: &mut ReaderBaseState| { &mut m.current_work },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<ReaderBaseState>(
                "ReaderBaseState",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static ReaderBaseState {
        static instance: ::protobuf::rt::LazyV2<ReaderBaseState> = ::protobuf::rt::LazyV2::INIT;
        instance.get(ReaderBaseState::new)
    }
}

impl ::protobuf::Clear for ReaderBaseState {
    fn clear(&mut self) {
        self.work_started = 0;
        self.work_finished = 0;
        self.num_records_produced = 0;
        self.current_work.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ReaderBaseState {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ReaderBaseState {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n+tensorflow/core/framework/reader_base.proto\x12\ntensorflow\"\xae\x01\
    \n\x0fReaderBaseState\x12!\n\x0cwork_started\x18\x01\x20\x01(\x03R\x0bwo\
    rkStarted\x12#\n\rwork_finished\x18\x02\x20\x01(\x03R\x0cworkFinished\
    \x120\n\x14num_records_produced\x18\x03\x20\x01(\x03R\x12numRecordsProdu\
    ced\x12!\n\x0ccurrent_work\x18\x04\x20\x01(\x0cR\x0bcurrentWorkBp\n\x18o\
    rg.tensorflow.frameworkB\x10ReaderBaseProtosP\x01Z=github.com/tensorflow\
    /tensorflow/tensorflow/go/core/framework\xf8\x01\x01b\x06proto3\
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