# Micrograd: every concept, as I'd answer it in an interview

**Source:** Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd* ([video](https://www.youtube.com/watch?v=VMj-3S1tku0), Zero to Hero lecture 1) · [micrograd on GitHub](https://github.com/karpathy/micrograd)

**How to read this:** each answer opens with what I'd say in the first 30 seconds. **Deeper** is what I add if they push. **Trap** is the mistake to avoid. Every number below was checked by re-running the lecture's examples.

---

## 1. The big picture

### What is backpropagation?

Backprop is how you get the gradient of a loss with respect to every parameter in a network, in one backward sweep. It is the chain rule from calculus, applied recursively from the output back to the inputs. Once I have those gradients, gradient descent nudges each weight a little in the direction that lowers the loss.

**Deeper:** backprop knows nothing about neural networks. It works on any expression built from differentiable operations. A neural network is just one kind of mathematical expression: the data and the weights go in, and predictions or a loss come out.

### What is an autograd engine? What is micrograd?

Autograd means automatic gradient. While you compute the forward pass, the engine records every operation in a graph. When you call `backward()`, it walks that graph in reverse and fills in the gradient of the output with respect to every node. Micrograd is a minimal version: about 100 lines for the engine and about 50 for a small neural network library on top.

### Why does micrograd work on single numbers? Isn't that slow?

It is very slow, on purpose. Breaking a network down to individual numbers and individual `+` and `×` makes every gradient visible. Real libraries pack those numbers into tensors (n-dimensional arrays) so that the same operations run in parallel on the hardware. The math does not change at all; tensors are only for speed. The lecture's claim, and I agree with it: micrograd is everything you need to train a network, and the rest is efficiency.

---

## 2. Derivatives

### What does a derivative tell you?

It is the sensitivity of the output to a tiny nudge of the input. If I increase `x` by a small `h`, how much does `f` move, and in which direction? It's rise over run as `h` goes to 0. The sign gives the direction and the size gives the strength.

Example: `f(x) = 3x² − 4x + 5`. At `x = 3`, `f = 20` and the slope is 14 (from `6x − 4`). At `x = −3` the slope is −22: nudging x up makes f go down. At `x = 2/3` the slope is 0, which is the bottom of the parabola. A small nudge there doesn't change the output.

### How do you estimate a derivative numerically?

`(f(x + h) − f(x)) / h` with a small `h`, like 0.001. It is an approximation that gets closer as `h` shrinks, until `h` is so small that floating-point rounding takes over and the answer gets worse again.

**Deeper:** the central difference `(f(x + h) − f(x − h)) / 2h` is more accurate. We don't train this way because it costs a forward pass per parameter: a million parameters means a million forward passes per step. Backprop gets every gradient for a small constant multiple of one forward pass. Numerical gradients are for checking, not training.

### And with several inputs?

You take a partial derivative for each input: nudge one input and hold the others fixed. For `d = a·b + c` at `a = 2, b = −3, c = 10` (so `d = 4`):

- `∂d/∂a = b = −3`. b is negative, so making a bigger makes d smaller.
- `∂d/∂b = a = 2`
- `∂d/∂c = 1`. c is added directly, so d moves one for one with it.

---

## 3. The computation graph

### How does an autograd engine remember what was computed?

Every number is wrapped in a `Value` object that stores:

- `data`: the number itself
- `grad`: the derivative of the final output with respect to this value
- `_prev`: the values that produced it, which are its children in the graph
- `_op`: the operation that produced it (`+`, `*`, `tanh`…)
- `_backward`: a small function that knows how to pass the gradient on to its children

Operator overloading (`__add__`, `__mul__`) means `a + b` builds a new `Value` and records where it came from. So the forward pass builds the graph for free.

**Deeper:** the graph is a DAG (directed acyclic graph). The leaves are the inputs and weights, and the root is the loss. The lecture draws it with Graphviz. The operation nodes in those drawings are only there to make the picture readable; they are not `Value` objects.

### What is the difference between the forward pass and the backward pass?

The forward pass computes the output from the inputs and keeps every intermediate value. The backward pass starts at the output with gradient 1 and pushes gradients back to every node. The backward pass needs values saved during the forward pass (for example, multiplication's gradient uses the other input's value). That is why training uses far more memory than inference.

---

## 4. Backprop by hand

### What does `.grad` on a node mean?

It is the derivative of the final output (the loss) with respect to that node. It starts at 0, meaning "no influence assumed yet". The output's own gradient is 1, because `dL/dL = 1`. That is the base case that starts backprop.

### What is the chain rule, and why is it the core of backprop?

If z depends on y and y depends on x, then `dz/dx = dz/dy · dy/dx`. Rates of change multiply. If A changes 3× as fast as B, and B changes 5× as fast as C, then A changes 15× as fast as C.

In backprop, each node only knows its **local derivative**: how its output responds to its own inputs. It knows nothing about the rest of the graph. The chain rule says the gradient for an input is the gradient arriving from above (`dL/d out`) times the local derivative. Do that at every node, from the output backwards, and you have every gradient:

```
input.grad += local_derivative * out.grad
```

### What does each operation do to the gradient?

| Operation | Local derivative | What it does to the gradient coming in from above |
|---|---|---|
| `a + b` | 1 for each input | Copies it unchanged to both inputs (a plus node is a router) |
| `a * b` | the other input | Each input gets the *other* input's value × the incoming gradient |
| `tanh(x)` | `1 − t²`, where `t` is the output | Scales it down. Close to 0 when the neuron is saturated |
| `exp(x)` | `eˣ`, which is the output itself | `out.data * out.grad` |
| `x ** k` (k a constant) | `k · x^(k−1)` | Power rule |

**Worked example:** `e = a·b`, `d = e + c`, `L = d·f`, with `a = 2, b = −3, c = 10, f = −2`, so `L = −8`.

- `dL/df = d = 4` and `dL/dd = f = −2`
- The plus node routes −2 unchanged to both `c` and `e`
- The times node: `dL/da = −2 × b = 6` and `dL/db = −2 × a = −4`

### How do you know your gradients are right?

A gradient check. Nudge one input by `h`, recompute the output, and compare `(L(x + h) − L(x)) / h` with the gradient backprop gave. They should match up to floating-point noise. I'd do this every time I write a new backward function. Micrograd's tests compare its forward and backward results against PyTorch for the same expressions.

### What does one optimization step look like?

To push `L` up, move each leaf a little along its gradient: `x += step * x.grad`. In the example above with `step = 0.01`, `L` goes from −8 to about −7.29. To lower a loss you step the other way, which is gradient descent.

---

## 5. A single neuron

### What is a neuron, mathematically?

A weighted sum of the inputs plus a bias, passed through a nonlinearity: `out = tanh(Σ wᵢxᵢ + b)`.

- The weights are like synapse strengths: each one multiplies its input.
- The bias sets how easily the neuron fires, whatever the input.
- The activation squashes the result. `tanh` maps everything into (−1, 1), with a smooth cap at each end.

**Deeper:** without the nonlinearity, stacked layers collapse into one linear function, and depth buys you nothing.

### What is the derivative of tanh?

`1 − tanh²(x)`. Handy, because `tanh(x)` is the output I already computed, so the backward pass is just `1 − t²`. In the lecture's neuron the input to tanh is 0.8814 and the output is 0.7071, so the local gradient is `1 − 0.5 = 0.5`.

### Walk me through backprop on one neuron.

The setup is `x1 = 2, x2 = 0, w1 = −3, w2 = 1, b ≈ 6.8814`, which gives an output of 0.7071.

1. `do/do = 1`, the base case.
2. tanh: `1 − 0.7071² = 0.5`.
3. The two plus nodes route 0.5 to `b`, `x1·w1` and `x2·w2`.
4. The times nodes: `w1.grad = x1 × 0.5 = 1.0`, `x1.grad = w1 × 0.5 = −1.5`, `w2.grad = x2 × 0.5 = 0`, `x2.grad = w2 × 0.5 = 0.5`.

**Why is `w2`'s gradient 0?** Because its input `x2` is 0. Changing `w2` changes nothing, so it has no influence on the output. A weight can only learn from inputs that are non-zero.

### Does it matter how coarse your operations are?

No. I can implement `tanh` as one operation with its own backward, or build it from `exp`, `+`, `−` and `/`, using `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`. You get the same output and the same gradients (I checked: both give 0.7071 and the same four gradients). An operation only needs a forward pass and a local derivative. Coarse operations give a smaller graph and are often faster and more numerically stable. Fine-grained ones mean less backward code to write. That is how PyTorch lets you add your own functions.

---

## 6. Making backward automatic

### How do you make backward automatic?

When each operation creates its output, it attaches a closure, `_backward`, that knows how to push `out.grad` into its inputs. Leaves get a function that does nothing. Then `backward()` on the loss:

1. Sorts the graph in topological order.
2. Sets `loss.grad = 1`.
3. Calls `_backward()` on every node in **reverse** topological order.

**Trap:** store the function, don't call it. Write `out._backward = _backward`, with no parentheses. Calling it stores `None`, and you get `'NoneType' object is not callable` later.

### Why do you need a topological sort?

A node can only pass its gradient on once it has received all of it, from every node that uses it. Topological order lists every node after all of its inputs. Reversed, it processes each node after everything that depends on it. You build it with a depth-first search: visit the children first, then add yourself to the list.

**Deeper:** a recursive DFS hits Python's recursion limit on very deep graphs. An iterative version avoids that.

### What goes wrong when a value is used more than once?

If a node feeds two places, the gradients from each use have to **add up**. With `b = a + a`, the right answer is `db/da = 2`. If the backward functions *set* `grad` with `=`, the second write overwrites the first and you get 1. The fix is to accumulate with `+=`. That is the multivariable chain rule: you sum over every path. It is also why gradients must start at 0.

---

## 7. More operations, and the Python plumbing

### How does `a + 1` work when 1 isn't a `Value`?

Wrap anything that isn't one: `other = other if isinstance(other, Value) else Value(other)`.

### Why does `2 * a` fail when `a * 2` works?

Python tries `int.__mul__(2, a)` first. That doesn't know about `Value`, so Python falls back to `a.__rmul__(2)`. Define `__rmul__` (and `__radd__`) to swap the operands and call the normal version.

### How do you get division and subtraction cheaply?

Build them from what you already have:

- `a / b = a · b⁻¹`, using a power operation with a constant exponent
- `a − b = a + (−b)`
- `−b = b · (−1)`

Only the power operation needs a new backward (the power rule). The exponent has to be a plain number; a `Value` exponent would need a different derivative.

---

## 8. PyTorch

### How does micrograd map to PyTorch?

Almost one to one: `Value` ↔ tensor, `.data` and `.grad` on both, `loss.backward()` on both. The differences:

- PyTorch works on tensors. Micrograd is the special case where every tensor holds one element.
- PyTorch defaults to float32, but Python floats are float64, so the lecture uses `.double()` to make the numbers match exactly.
- Leaf tensors default to `requires_grad=False`, because you usually don't need gradients for the inputs. You turn it on for the parameters.
- `.item()` takes the plain number out of a one-element tensor.

Running the same neuron in PyTorch gives the same 0.7071 and the same gradients.

### Where is tanh's backward inside PyTorch?

It's there, but buried in separate CPU and CUDA kernels with branches for different data types. The core line is still `grad × (1 − y²)`. Production libraries are built to be used, not read, which is why micrograd is worth having as a learning tool.

### How would you add your own operation to PyTorch?

Subclass `torch.autograd.Function` and write a static `forward` and `backward`. If you can supply the local gradient, autograd chains it into everything else.

---

## 9. Building the network

### How are Neuron, Layer and MLP put together?

- **Neuron:** a list of weights and a bias, each random in [−1, 1]. Calling it on `x` returns `tanh(w·x + b)`.
- **Layer:** a list of independent neurons that all see the same input.
- **MLP (multi-layer perceptron):** layers in sequence; each layer's outputs are the next layer's inputs.

The API copies `torch.nn.Module`: modules are callable, and `parameters()` returns every weight and bias so the training loop can update them all.

### How many parameters does a 3 → [4, 4] → 1 MLP have?

`(3×4 + 4) + (4×4 + 4) + (4×1 + 1) = 16 + 20 + 5 = 41`. Each layer has `n_in × n_out` weights plus `n_out` biases.

---

## 10. Loss and training

### What is a loss function, and why is it a single number?

It's one number that measures how badly the network does on the data. Optimization needs one scalar to minimize, and backprop from that scalar gives every parameter its gradient. You design it so that a low loss means the network is doing what you want.

### What loss does the lecture use, and why squared error?

`Σ (prediction − target)²` over the four examples. Squaring removes the sign, so overshooting and undershooting both count. It's zero only when the prediction is exact, and big misses cost much more than small ones (an absolute value would also remove the sign). The lecture sums instead of averaging. Taking the mean only rescales the gradient, which has the same effect as changing the learning rate.

### Do you use the gradients on the input data?

Backprop computes them, but training doesn't use them. The data is fixed, so we only update the parameters.

### Why the minus sign in the update?

The gradient points in the direction where the loss *increases* fastest. To minimize, step the other way: `p.data -= lr * p.grad`. Sanity check: a weight with a negative gradient lowers the loss when it increases, and the minus sign makes it increase.

### How do you choose the learning rate?

Too small and it converges slowly. Too large and it overshoots: the loss bounces around or blows up, because the gradient only describes the loss very close to where you are. Try a few values on a log scale, watch the loss, and lower the rate as training goes on (decay).

### What does the training loop look like?

```python
for step in range(steps):
    loss = sum((mlp(x) - y) ** 2 for x, y in zip(xs, ys))   # 1. forward
    for p in mlp.parameters():
        p.grad = 0.0                                        # 2. zero the grads
    loss.backward()                                         # 3. backward
    for p in mlp.parameters():
        p.data -= lr * p.grad                               # 4. update
```

### What is the zero-grad bug?

Gradients accumulate (`+=`), so if you don't reset them, each step uses the sum of all previous gradients. That acts like a growing, uncontrolled step size. In the lecture it happened to converge *faster*, because the problem is tiny. Re-running the same four-example setup for 20 steps (lr 0.05, seed 1337) gives a loss of 6.0 → 0.06 with zero-grad and 6.0 → 4.0 without it. The bug can look like working code, which is why I'd want to know it cold.

**Deeper: why does PyTorch accumulate instead of overwriting?** So you can sum gradients over several backward calls: gradient accumulation across micro-batches when a full batch doesn't fit in memory, or several losses. The price is that you must reset them yourself with `optimizer.zero_grad()`.

---

## 11. From micrograd to GPT

### How does this relate to something like GPT?

It's the same loop at a huge scale. GPT-style models have billions of parameters, but training is still forward → loss → backward → update. What changes:

- The loss is **cross-entropy** on predicting the next token, not squared error.
- Everything is tensors on GPUs.
- The update rule is fancier than plain gradient descent (Adam and AdamW in practice).
- Each step uses a random **minibatch** instead of the whole dataset.

### What else does micrograd's own demo add?

- **Minibatches:** each step uses a random subset of the data.
- **Max-margin (hinge) loss** for binary classification. Binary cross-entropy and squared error also work on simple problems.
- **L2 regularization:** a penalty on large weights, which reduces overfitting.
- **Learning-rate decay:** big steps early, small steps near the end.
- **ReLU** instead of tanh: `max(0, x)`, with gradient 1 when the input is positive and 0 otherwise.

---

## 12. Follow-ups I should expect (beyond the lecture)

- **Forward-mode vs reverse-mode autodiff.** Forward mode costs one pass per *input*; reverse mode (backprop) costs about one pass per *output*. A loss is one number with millions of inputs (the parameters), so reverse mode wins.
- **Vanishing gradients.** When tanh saturates (output near ±1), `1 − t²` is close to 0 and the gradient dies. Across many layers these small factors multiply together. Fixes: ReLU-style activations, careful initialization, normalization layers, residual connections.
- **Memory.** The backward pass needs the activations saved during the forward pass. Gradient checkpointing saves memory by recomputing some of them during the backward pass.
- **Squared error for classification.** With saturating outputs, it gives tiny gradients exactly when the model is confidently wrong. Cross-entropy's gradient on the logits is `p − y`, which stays large when the model is wrong.
- **In-place operations.** Changing a value that the backward pass still needs corrupts the gradients. PyTorch raises an error when it detects this.

---

## 13. Rapid fire

| Question | One line |
|---|---|
| Backprop in one sentence? | The chain rule applied backwards through the computation graph to get every gradient in one pass. |
| What does `x.grad` mean? | How much the final loss changes per unit nudge of `x`. |
| Why start the output at 1? | `dL/dL = 1`, the base case. |
| What does a plus node do to the gradient? | Copies it unchanged to all of its inputs. |
| What does a times node do? | Each input gets the other input's value × the incoming gradient. |
| tanh's derivative? | `1 − t²`, using the output `t`. |
| Why `+=` instead of `=`? | A value used in several places gets the sum of the gradients from every use. |
| Why a topological sort? | So a node runs its backward only after all of its gradient has arrived. |
| Why zero the grads each step? | Otherwise they keep adding up from previous steps. |
| Why `−lr × grad`? | The gradient points uphill and we want to go downhill. |
| Why tensors in real libraries? | Speed through parallelism. The math is identical. |
| Parameters in 3 → [4, 4] → 1? | 41. |

---

## 14. Whiteboard version

The core engine is small enough to write from memory. Type it out in `../01-micrograd/` without looking.

```python
import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data, self.grad = data, 0.0
        self._backward = lambda: None
        self._prev, self._op = set(_children), _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```
