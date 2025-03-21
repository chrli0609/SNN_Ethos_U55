#include "third_party/tensorflow/lite/c/c_api.h"
#include "third_party/tensorflow/lite/c/c_api_opaque.h"


TfLiteRegistrationExternal*
TfLiteRegistrationExternalCreate(
    TfLiteBuiltinOperator builtin_code,  // Normally `TfLiteBuiltinCustom`.
    const char* custom_name,  // The name of the custom op.
    int version  // Normally `1` for the first version of a custom op.
);



// Initializes the op from serialized data.
void* Init(TfLiteOpaqueContext* context, const char* buffer, size_t length);

// Deallocates the op.
// The pointer `buffer` is the data previously returned by an Init invocation.
void Free(TfLiteOpaqueContext* context, void* buffer);

// Called when the inputs that this node depends on have been resized.
TfLiteStatus Prepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

// Called when the node is executed. (Should read node inputs and write to
// node outputs).
TfLiteStatus Invoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node);

// Retrieves the async kernel.
TfLiteAsyncKernel AsyncKernel(TfLiteOpaqueContext* context,
                              TfLiteOpaqueNode* node);



void* MyCustomOpInit(TfLiteOpaqueContext* context,
                     const char* buffer, size_t length) { ... }
// ... plus definitions of MyCustomOpFree, MyCustomOpPrepare, and
// MyCustomOpInvoke.
      



void TfLiteRegistrationExternalSetInit(
    TfLiteRegistrationExternal* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length));
void TfLiteRegistrationExternalSetFree(
    TfLiteRegistrationExternal* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data));
void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));
void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));
void TfLiteRegistrationExternalSetAsyncKernel(
    TfLiteRegistrationExternal* registration,
    struct TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node));