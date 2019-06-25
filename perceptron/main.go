package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Neuron struct {
	weights []float64
}

func (n *Neuron) init(inputsCount int) {
	rand.Seed(time.Now().Unix())
	n.weights = make([]float64, inputsCount)
	for i, _ := range n.weights {
		n.weights[i] = rand.Float64()*2 - 1
	}
}

func (n *Neuron) think(inputs []float64) float64 {
	var sum float64 = 0
	for i, w := range n.weights {
		sum += w * inputs[i]
	}
	return sigmoid(sum)
}

func (n *Neuron) train(trainingSetInputs [][]float64, trainingSetOutputs []float64, iterations int) {
	for i := 0; i < iterations; i++ {
		adjustments := make([]float64, len(n.weights))

		for setIdx, inputs := range trainingSetInputs {
			output := n.think(inputs)
			err := trainingSetOutputs[setIdx] - output

			for col := 0; col < len(n.weights); col++ {
				adjustments[col] += trainingSetInputs[setIdx][col] * err * sigmoidDerivative(output)
			}
		}

		for i := 0; i < len(n.weights); i++ {
			n.weights[i] += adjustments[i]
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func main() {
	var n Neuron
	n.init(3)
	fmt.Printf("Start Weights: %v\n", n.weights)

	trainingSetInputs := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}
	trainingSetOutputs := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	test := []float64{0, 1, 1}
	var res float64

	n.train(trainingSetInputs, trainingSetOutputs, 50000)
	fmt.Printf("Weights: %v\n", n.weights)
	res = n.think(test)
	fmt.Printf("Result: %f\n", res)
}
