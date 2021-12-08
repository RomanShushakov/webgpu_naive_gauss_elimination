use wasm_bindgen::prelude::*;
// use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
// use wasm_bindgen_futures::spawn_local;
use web_sys::
{
    GpuDevice, GpuBufferDescriptor, GpuBufferUsage, GpuShaderModuleDescriptor, 
    GpuProgrammableStage, GpuComputePipelineDescriptor, GpuBufferBinding, GpuBindGroupEntry, 
    GpuBindGroupDescriptor, GpuMapMode,

    // GpuAdapter, 
    // Window, WorkerGlobalScope, WorkerNavigator,
};
use js_sys::{Float32Array, Uint32Array};


// #[wasm_bindgen]
// extern "C"
// {
//     #[wasm_bindgen(js_namespace = console)]
//     pub fn log(value: &str);
// }


// async fn device() -> Result<GpuDevice, JsValue>
// {
//     let window = web_sys::window().expect("There are no window!");
//     log("Window was selected");
//     let navigator = window.navigator();
//     log("Navigator was selected");
//     let gpu = navigator.gpu();
//     log("Gpu was selected");
//     let gpu_adapter_value =  JsFuture::from(gpu.request_adapter()).await?;
//     let gpu_adapter = GpuAdapter::from(gpu_adapter_value);
//     log("Adapter was selected");
//     let gpu_device_value = JsFuture::from(gpu_adapter.request_device()).await?;
//     let gpu_device = GpuDevice::from(gpu_device_value);
//     log("Device was selected");
//     Ok(gpu_device)
// }


// async fn worker_device() -> Result<GpuDevice, JsValue>
// {
    
//     let global = js_sys::global();
//     let obj = 
//     {
//         if js_sys::eval("typeof WorkerGlobalScope !== 'undefined'")?.as_bool().unwrap() 
//         {
//             Some(global.dyn_into::<WorkerGlobalScope>()?)
//         }
//         else 
//         {
//             None
//         }
//     };

//     if let Some(worker_global_scope) = obj 
//     {
//         let navigator = worker_global_scope.navigator();
//         let gpu = navigator.gpu();
//         log("Gpu was selected");
//         let gpu_adapter_value =  JsFuture::from(gpu.request_adapter()).await?;
//         let gpu_adapter = GpuAdapter::from(gpu_adapter_value);
//         log("Adapter was selected");
//         let gpu_device_value = JsFuture::from(gpu_adapter.request_device()).await?;
//         let gpu_device = GpuDevice::from(gpu_device_value);
//         log("Device was selected");
//         return Ok(gpu_device);
//     }
//     else
//     {
//         return Err(JsValue::from("There are no worker global scope"));    
//     }
// }


#[wasm_bindgen]
pub async fn naive_gauss_elimination(gpu_device: GpuDevice, 
    a_rows_number: u32, a_columns_number: u32, a_elements_values: Vec<f32>,
    b_elements_values: Vec<f32>) -> Result<JsValue, JsValue>
{
    // let gpu_device = device().await?;
    // let gpu_device = worker_device().await?;

    let a_size = (a_elements_values.len() * std::mem::size_of::<f32>()) as f64;
    let a_usage = GpuBufferUsage::STORAGE | GpuBufferUsage::COPY_SRC;
    let mut gpu_buffer_a_descriptor = GpuBufferDescriptor::new(a_size, a_usage);
    let gpu_buffer_a = gpu_device.create_buffer(gpu_buffer_a_descriptor.mapped_at_creation(true));
    let a_array_buffer = gpu_buffer_a.get_mapped_range();
    Float32Array::new(&a_array_buffer).copy_from(&a_elements_values);
    drop(a_array_buffer);
    gpu_buffer_a.unmap();


    let a_shape_size = (2 * std::mem::size_of::<u32>()) as f64;
    let a_shape_usage = GpuBufferUsage::UNIFORM;
    let mut gpu_buffer_a_shape_descriptor = GpuBufferDescriptor::new(a_shape_size, a_shape_usage);
    let gpu_buffer_a_shape = gpu_device.create_buffer(gpu_buffer_a_shape_descriptor.mapped_at_creation(true));
    let a_shape_array_buffer = gpu_buffer_a_shape.get_mapped_range();
    Uint32Array::new(&a_shape_array_buffer).copy_from(&[a_rows_number, a_columns_number]);
    drop(a_shape_array_buffer);
    gpu_buffer_a_shape.unmap();


    let b_size = (b_elements_values.len() * std::mem::size_of::<f32>()) as f64;
    let b_usage = GpuBufferUsage::STORAGE | GpuBufferUsage::COPY_SRC;
    let mut gpu_buffer_b_descriptor = GpuBufferDescriptor::new(b_size, b_usage);
    let gpu_buffer_b = gpu_device.create_buffer(gpu_buffer_b_descriptor.mapped_at_creation(true));
    let b_array_buffer = gpu_buffer_b.get_mapped_range();
    Float32Array::new(&b_array_buffer).copy_from(&b_elements_values);
    drop(b_array_buffer);
    gpu_buffer_b.unmap();


    let init_iteration = 1u32;

    let iteration_number_size = (1 * std::mem::size_of::<u32>()) as f64;
    let iteration_number_usage = GpuBufferUsage::STORAGE;
    let mut gpu_buffer_iteration_number_descriptor = GpuBufferDescriptor::new(iteration_number_size, iteration_number_usage);
    let gpu_buffer_iteration_number = gpu_device.create_buffer(gpu_buffer_iteration_number_descriptor.mapped_at_creation(true));
    let iteration_number_array_buffer = gpu_buffer_iteration_number.get_mapped_range();
    Uint32Array::new(&iteration_number_array_buffer).copy_from(&[init_iteration]);
    drop(iteration_number_array_buffer);
    gpu_buffer_iteration_number.unmap();


    let gpu_shader_module_descriptor = GpuShaderModuleDescriptor::new(include_str!("../shaders/compute.wgsl"));
    let gpu_shader_module = gpu_device.create_shader_module(&gpu_shader_module_descriptor);

    let gpu_programmable_stage = GpuProgrammableStage::new("main", &gpu_shader_module);
    let gpu_compute_pipeline_descriptor = GpuComputePipelineDescriptor::new(&gpu_programmable_stage);
    let gpu_compute_pipeline = gpu_device.create_compute_pipeline(&gpu_compute_pipeline_descriptor);


    let gpu_buffer_binding_a = GpuBufferBinding::new(&gpu_buffer_a);
    let gpu_bind_group_entry_a = GpuBindGroupEntry::new(0u32, &gpu_buffer_binding_a);


    let gpu_buffer_binding_a_shape = GpuBufferBinding::new(&gpu_buffer_a_shape);
    let gpu_bind_group_entry_a_shape = GpuBindGroupEntry::new(1u32, &gpu_buffer_binding_a_shape);


    let gpu_buffer_binding_b = GpuBufferBinding::new(&gpu_buffer_b);
    let gpu_bind_group_entry_b = GpuBindGroupEntry::new(2u32, &gpu_buffer_binding_b);


    let gpu_buffer_binding_iteration_number = GpuBufferBinding::new(&gpu_buffer_iteration_number);
    let gpu_bind_group_entry_iteration_number = GpuBindGroupEntry::new(3u32, &gpu_buffer_binding_iteration_number);


    let gpu_bind_group_entries = [gpu_bind_group_entry_a, gpu_bind_group_entry_a_shape,
        gpu_bind_group_entry_b, gpu_bind_group_entry_iteration_number].iter().collect::<js_sys::Array>();

    let gpu_bind_group_layout = gpu_compute_pipeline.get_bind_group_layout(0u32);
    let gpu_bind_group_descriptor = GpuBindGroupDescriptor::new(&gpu_bind_group_entries, &gpu_bind_group_layout);
    let gpu_bind_group = gpu_device.create_bind_group(&gpu_bind_group_descriptor);

    let gpu_command_encoder = gpu_device.create_command_encoder();
    let gpu_compute_pass_encoder = gpu_command_encoder.begin_compute_pass();
    gpu_compute_pass_encoder.set_pipeline(&gpu_compute_pipeline);
    gpu_compute_pass_encoder.set_bind_group(0u32, &gpu_bind_group);

    for _ in init_iteration..a_rows_number * 2
    {
        gpu_compute_pass_encoder.dispatch(a_rows_number);
    }

    gpu_compute_pass_encoder.end_pass();
    
    let read_usage = GpuBufferUsage::COPY_DST | GpuBufferUsage::MAP_READ;
    let gpu_buffer_read_descriptor = GpuBufferDescriptor::new(b_size, read_usage);
    let gpu_buffer_read = gpu_device.create_buffer(&gpu_buffer_read_descriptor);
    gpu_command_encoder.copy_buffer_to_buffer_with_f64_and_f64_and_f64(
        &gpu_buffer_b, 
        0.0,
        &gpu_buffer_read,
        0.0,
        b_size,
    );

    let gpu_commands = gpu_command_encoder.finish();
    let gpu_commands_composed = [gpu_commands].iter().collect::<js_sys::Array>();
    gpu_device.queue().submit(&gpu_commands_composed);
    JsFuture::from(gpu_buffer_read.map_async(GpuMapMode::READ)).await?;
    let array_buffer = gpu_buffer_read.get_mapped_range();

    Ok(array_buffer.into())
}
