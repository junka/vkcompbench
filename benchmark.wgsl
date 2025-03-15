struct Buffer {
  x: array<vec4f>
};

@group(0) @binding(0) var<storage, read> inputA: Buffer;
@group(0) @binding(1) var<storage, read> inputB: Buffer;
@group(0) @binding(2) var<storage, read_write> outputO: Buffer;

@id(0) override kLoopSize: u32 = 10000u;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx: u32 = global_id.x;
    var a: vec4f = inputA.x[idx];
    var b: vec4f = inputB.x[idx];
    var c: vec4f = vec4f(1.0, 1.0, 1.0, 1.0);

    for (var i: u32 = 0u; i < kLoopSize; i = i + 1u) {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    outputO.x[idx] = c;
}
