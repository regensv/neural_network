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

func (n *Neuron) activation(x float64) float64 {
	// return sigmoid(x)
	return x
}

func (n *Neuron) think(inputs []float64) float64 {
	var sum float64 = 0
	for i, w := range n.weights {
		sum += w * inputs[i]
	}
	return n.activation(sum)
}

func (n *Neuron) train(trainingSetInputs [][]float64, trainingSetOutputs []float64, iterations int) {
	for i := 0; i < iterations; i++ {
		for setIdx, inputs := range trainingSetInputs {
			output := n.think(inputs)
			err := trainingSetOutputs[setIdx] - output

			for col := 0; col < len(n.weights); col++ {
				n.weights[col] += trainingSetInputs[setIdx][col] * err / 1000
			}
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
	n.init(2)
	fmt.Printf("Start Weights: %v\n", n.weights)

	trainingSetInputs := [][]float64{
		{1, 1},
		{2, 1},
		{3, 1},
		{4, 1},
		{5, 1},
	}
	trainingSetOutputs := []float64{9, 12, 15, 18, 21}

	test := []float64{13, 1}
	var res float64

	n.train(trainingSetInputs, trainingSetOutputs, 10000)
	res = n.think(test)
	fmt.Printf("Result: %f, Weights: %v\n", res, n.weights)
}
