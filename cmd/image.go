package main

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
)

const imageSize = 299
const maxImagePixels = 16_000_000
const maxImageDimension = 8192

var errImageTooLarge = errors.New("image dimensions exceed the limit")

type imageTensor [][][3]float32

// normalizeImage follows the original ResizeBilinear defaults: align_corners=false,
// half_pixel_centers=false, then RGB / 255. DecodeConfig bounds memory before decode.
func normalizeImage(data []byte) (imageTensor, error) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || (format != "jpeg" && format != "png") {
		return nil, errors.New("expected a JPEG or PNG image")
	}
	if config.Width <= 0 || config.Height <= 0 || config.Width > maxImageDimension || config.Height > maxImageDimension || int64(config.Width)*int64(config.Height) > maxImagePixels {
		return nil, errImageTooLarge
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, errors.New("invalid image data")
	}
	bounds := img.Bounds()
	result := make(imageTensor, imageSize)
	xScale, yScale := float32(bounds.Dx())/imageSize, float32(bounds.Dy())/imageSize
	rgb := func(x, y int) [3]float32 {
		// NRGBA conversion retains unassociated RGB for PNGs with transparency.
		c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
		return [3]float32{float32(c.R), float32(c.G), float32(c.B)}
	}
	for y := range result {
		result[y] = make([][3]float32, imageSize)
		sy := float32(y) * yScale
		y0 := min(int(sy), bounds.Dy()-1)
		y1 := min(y0+1, bounds.Dy()-1)
		dy := sy - float32(y0)
		for x := range result[y] {
			sx := float32(x) * xScale
			x0 := min(int(sx), bounds.Dx()-1)
			x1 := min(x0+1, bounds.Dx()-1)
			dx := sx - float32(x0)
			a, b, c, d := rgb(x0, y0), rgb(x1, y0), rgb(x0, y1), rgb(x1, y1)
			for ch := range 3 {
				top := a[ch] + (b[ch]-a[ch])*dx
				bottom := c[ch] + (d[ch]-c[ch])*dx
				result[y][x][ch] = (top + (bottom-top)*dy) / 255
			}
		}
	}
	return result, nil
}
