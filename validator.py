class Val():
    def __init__(self, model_layout) -> None:
        self.model_layout= model_layout

    def check_model(self, shape=448):
        shapes = [shape]
        for i in range(len(self.model_layout)):
            layer = self.model_layout[i]
            shape = self._calc_dim(shape, layer[2], layer[3], layer[4])
            shapes.append(shape)
        print(shapes)

    def _calc_dim(self, input, kernel, stride, padding):
        padding_top, padding_bottom = padding, padding
        output_h = (input + padding_top + padding_bottom - kernel) / (stride) + 1
        return round(output_h)

    def _calc_padding(self, input, output, kernel, stride):
        padding  = ((output-1) * stride) + kernel - input
        return round(padding/2)
