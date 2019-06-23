package main

import (
	"fmt"
	"math"
)

func sigmoid(v float64) float64 {
	return 1 / (1 + math.Pow(math.E, -v))
}

func activation(v float64) float64 {
	return sigmoid(v)
}

type Neuron struct {
	w     float64
	val   float64
	prev  []*Neuron
	delta float64
	wGrad float64
}

func (n *Neuron) getValue() float64 {
	if len(n.prev) != 0 {
		var val float64 = 0
		for _, in := range n.prev {
			val += in.getValue() * in.w
		}
		n.val = activation(val)
	}

	// fmt.Printf("weight: %f, value: %f\n", n.w, n.val)
	return n.val
}

func (n *Neuron) UpdateW(delta float64) {
	var E float64 = 0.7 // learn speed
	var A float64 = 0.1 // moment

	deltaNew := ((1 - n.val) * n.val) * (n.w * delta)
	n.wGrad = n.val * delta
	deltaW := E*n.wGrad + A*n.delta
	n.delta = deltaNew
	n.w += deltaW

	for _, n := range n.prev {
		n.UpdateW(n.delta)
	}
}

func think(nIn1 *Neuron, nIn2 *Neuron, nOut *Neuron, in1 float64, in2 float64) float64 {
	nIn1.val = in1
	nIn2.val = in2
	result := nOut.getValue()
	return result
}

func train(nIn1 *Neuron, nIn2 *Neuron, nOut *Neuron, sets [][]float64) {
	var setsNum int = 0
	var errSum float64 = 0

	for i := 0; i < len(sets); i++ {
		result := think(nIn1, nIn2, nOut, sets[i][0], sets[i][1])
		// fmt.Printf("set %d. Result: %f\n", i, result)

		diff := sets[i][2] - result
		fmt.Printf("result: %f, diff: %f\n", result, diff)
		errSum += diff * diff
		setsNum++

		nOut.delta = diff * ((1 - result) * result)
		for _, n := range nOut.prev {
			n.UpdateW(nOut.delta)
		}
	}

	err := errSum / float64(setsNum)

	fmt.Printf("Epoch error: %f\n", err)
}

func main() {
	nI1 := Neuron{0.45, 0.5, []*Neuron{}, 0.0, 0.0}
	nI2 := Neuron{-0.12, 0.4, []*Neuron{}, 0.0, 0.0}

	nH1 := Neuron{1.5, 0.0, []*Neuron{
		&nI1,
		&nI2,
	}, 0.0, 0.0}

	nO1 := Neuron{0.0, 0.0, []*Neuron{
		&nH1,
	}, 0.0, 0.0}

	var iterations int = 10

	trainingData := [][]float64{
		[]float64{0.5, 0.4, 0.9},
		[]float64{0.1, 0.1, 0.2},
		[]float64{0.3, 0.1, 0.4},
		[]float64{0.7, 0.05, 0.75},
	}

	for i := 0; i < iterations; i++ {
		train(&nI1, &nI2, &nO1, trainingData)
	}

	result := think(&nI1, &nI2, &nO1, 0.1, 0.1)
	fmt.Printf("result: %f\n", result)

	// var guess float64
	// guess = n.think(0.5, 0.4)
	// fmt.Printf("Guessed: %f\n", guess)
	// guess = n.think(0.1, 0.3)
	// fmt.Printf("Guessed: %f\n", guess)
	// guess = n.think(0.6, 0.2)
	// fmt.Printf("Guessed: %f\n", guess)

	// 	fmt.Printf("weights: (%f, %f)\n", nn.w1, nn.w2)

	// 	guess := nn.think(2, 3)
	// 	fmt.Printf("Guessed: %f\n", guess)
	// 	guess = nn.think(5, 4)
	// 	fmt.Printf("Guessed: %f\n", guess)
	// 	guess = nn.think(0, 0)
	// 	fmt.Printf("Guessed: %f\n", guess)
	// 	guess = nn.think(12, 3)
	// 	fmt.Printf("Guessed: %f\n", guess)
	// 	guess = nn.think(32, 17)
	// 	fmt.Printf("Guessed: %f\n", guess)
}
