import init, { naive_gauss_elimination } from "../wasm/gauss_elimination/gauss_elimination.js";

await init();


const a_rows_number = 3;
const a_columns_number = 3;

const a_elements_values = new Float32Array([
    3.0, -0.1, -0.2,
    0.1, 7.0, -0.3,
    0.3, -0.2, 10.0,
]);


const b_elements_values = new Float32Array([
    7.85,
    -19.3,
    71.4,
]);


// const a_rows_number = 10;
// const a_columns_number = 10;

// const a_elements_values = [
//     2225275.0, 803571.6, 0.0, 0.0, 0.0, 247252.78, 61813.16, 0.0, 0.0, 0.0, 
//     803571.6, 2225275.0, 0.0, 0.0, 0.0, -61813.164, -1359890.4, 0.0, 0.0, 0.0, 
//     0.0, 0.0, 961500.1, -2403750.3, 2403750.3, 0.0, 0.0, -240375.03, -2403750.3, 1201875.1, 
//     0.0, 0.0, -2403750.3, 12022923.0, -1506.6967, 0.0, 0.0, 2403750.3, 12016202.0, 115.899704, 
//     0.0, 0.0, 2403750.3, -1506.6967, 12022923.0, 0.0, 0.0, 1201875.1, -115.89969, 6009839.0, 
//     247252.78, -61813.164, 0.0, 0.0, 0.0, 2225275.0, -803571.56, 0.0, 0.0, 0.0, 
//     61813.16, -1359890.4, 0.0, 0.0, 0.0, -803571.56, 2225275.0, 0.0, 0.0, 0.0, 
//     0.0, 0.0, -240375.03, 2403750.3, 1201875.1, 0.0, 0.0, 961500.1, 2403750.3, 2403750.0, 
//     0.0, 0.0, -2403750.3, 12016202.0, -115.89969, 0.0, 0.0, 2403750.3, 12022924.0, 1506.6967, 
//     0.0, 0.0, 1201875.3, 115.899704, 6009839.5, 0.0, 0.0, 2403750.0, 1506.6967, 12022924.0
// ];


// const b_elements_values = [
//     300.0, 
//     0.0, 
//     0.5, 
//     0.0, 
//     0.0, 
//     300.0, 
//     0.0, 
//     0.5, 
//     0.0, 
//     0.0
// ];


const button = document.querySelector(".click");
button.addEventListener("click", async () => {

    if (!navigator.gpu) {
        console.log("WebGPU is not supported.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.log("Failed to get GPU adapter.");
        return;
    }
    const device = await adapter.requestDevice();

    naive_gauss_elimination(device, a_rows_number, a_columns_number, a_elements_values, b_elements_values)
        .then((result) => console.log(new Float32Array(result)));
    console.log("click");
});
