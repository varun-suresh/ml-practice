from random import uniform
import math
from nn import Module
from engine import Value

class Conv1D(Module):
    def __init__(self, inputChannels, outputChannels, kernelSize, padding, stride=1, useBias=False):
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.padding = padding if padding in ("same","valid") else None
        self.useBias = useBias
        self.stride = stride
        if self.stride != 1 and self.padding == "same":
            raise Exception("Padding can be set to same only if stride is 1")
        self._kernel = [[[Value(uniform(-1,1)) for _ in range(kernelSize)] for _ in range(self.inputChannels)] for _ in range(self.outputChannels)]
        if self.useBias:
            self._bias = [Value(0) for _ in range(self.outputChannels)]

    def parameters(self):
        params = []
        for i in range(self.outputChannels):
            for j in range(self.inputChannels):
                for param in self._kernel[i][j]:
                    params.append(param)
            if self.useBias:
                params.append(self._bias[i])

        return params
    
    def _padInput(self, input):
        paddedInput = []
        for inputChannel in range(self.inputChannels):
            paddedInputChannel = []
            [paddedInputChannel.append(0) for _ in range(math.ceil((self.kernelSize - 1)/2))]
            [paddedInputChannel.append(i) for i in input[inputChannel]]
            if self.kernelSize % 2 == 0:
                [paddedInputChannel.append(0) for _ in range(math.ceil((self.kernelSize - 1)/2)-1)]
            else:
                [paddedInputChannel.append(0) for _ in range(math.ceil((self.kernelSize - 1)/2))]   
            paddedInput.append(paddedInputChannel)
         
        return paddedInput
        

    def __call__(self, input):
        outChannel = []
        if self.padding == "same":
            input = self._padInput(input)

        inputSize = len(input[0])
        for outputChannel in range(self.outputChannels):
            kernelOutput = []
            for i in range(0, inputSize - self.kernelSize + 1, self.stride):
                out = 0
                for j in range(self.kernelSize):
                    for inputChannel in range(self.inputChannels):
                        out += self._kernel[outputChannel][inputChannel][j] * input[inputChannel][i+j]
                if self.useBias:
                    out += self._bias[outputChannel]
                kernelOutput.append(out)
            outChannel.append(kernelOutput)
            
        return outChannel

if __name__ == "__main__":
    conv1 = Conv1D(1,1,3,"valid",useBias=False)
    out = conv1([[1,2,3,4]])
    print(len(out[0]))
     