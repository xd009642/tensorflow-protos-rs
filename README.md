# Tensorflow Protos Rust

This crate contains tensorflow protobuf files compiled into Rust via 
[protobuf-rust](https://github.com/stepancheg/rust-protobuf) and
associated crates.

There are a lot of protobuf files in tensorflow and it can be hard to
track which ones you need manually. The aim of this crate as a start
will be to filter out unnecessary protobufs i.e. python/example/tests
and potentially in future apply feature gates to ensure you bring in
the minimum required for your needs.

Currently I am only targetting tensorflow 1.15. This is because without
the C bindings being published for 2.x the same as 1.x it's not feasible
to upgrade tensorflow rust code to 2.x at this time.

I'm also ignoring tensorflow lite for now as I don't use it. There will
also have to be a bit of extra work for it because of duplicate names
(types.proto).

## License

For simplicity, this work is licensed the same as tensorflow under the 
Apache License (version 2). See LICENSE for more details.
