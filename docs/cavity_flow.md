# Steady Channel Flow (Cavity)

This notebook demonstrates solving steady-state Cavity Flow with the same setup as [this Comsol blog](https://www.comsol.com/blogs/how-to-solve-a-classic-cfd-benchmark-the-lid-driven-cavity-problem).

## 1. Define Geometry Domain

Set up the computational domain.

```python
import deepflow as df
print("Deepflow is runned on:", df.device) # to change to cpu use df.device = 'cpu'
df.manual_seed(69) # for reproducibility

rectangle = df.geometry.rectangle([0, 1], [0, 1])
domain = df.domain(rectangle)
domain.show_setup()
```

*(Output skipped)*

## 2. Define Physics

Define the Navier-Stokes equations for fluid flow and apply boundary conditions (e.g., no-slip walls, inlet velocity).

```python
domain.bound_list[0].define_bc({'u': 0,'v': 0})
domain.bound_list[1].define_bc({'u': 0,'v': 0})
domain.bound_list[2].define_bc({'u': 0,'v': 0})
domain.bound_list[3].define_bc({'u': 1, 'v': 0})
domain.area_list[0].define_pde(df.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000))
domain.show_setup()
```

Sample initial points for training.

```python
domain.sampling_lhs(bound_sampling_res=[1000, 1000, 1000, 1000], area_sampling_res=[2000])
domain.show_coordinates(display_physics=False)
```

## 3. Train the PINN model

Define how collocation points are sampled during training.

```python
def do_in_adam(epoch, model):
    return
        
def do_in_lbfgs(epoch, model):
    if epoch % 100 == 0 and epoch > 0:
        domain.sampling_R3(bound_sampling_res=[1000, 1000, 1000, 1000], area_sampling_res=[2000])
        print(domain)
```

Train the model using Adam for initial training (faster convergence).

```python
model0 = df.PINN(width=50, length=5, input_vars=['x','y'], output_vars=['u','v','p'])

# Train the model
model1, model1best = model0.train_adam(
    learning_rate=0.004,
    epochs=2000,
    calc_loss=df.calc_loss_simple(domain),
    threshold_loss=0.05,
    do_between_epochs=do_in_adam)
```

Refine the model using LBFGS for higher precision.

```python
# Train the model
model2 = model1best.train_lbfgs(
    calc_loss=df.calc_loss_simple(domain),
    epochs=450,
    threshold_loss=0.0005,
    do_between_epochs=do_in_lbfgs)
```

## 4. Visualization

### 4.1 Visualize area

```python
df.Visualizer.refwidth_default = 4

# Create object for evaluation
area_eval = domain.area_list[0].evaluate(model2)
# Sampling uniform points
area_eval.sampling_area([200, 200])

area_eval['v_mag'] = (area_eval['u']**2 + area_eval['v']**2)**0.5

_ = area_eval.plot_color('v_mag', s=4, cmap='jet')
_ = area_eval.plot_color('u', s=4, cmap='rainbow')
_ = area_eval.plot_color('v', s=4, cmap='rainbow')
_ = area_eval.plot_color('p', s=4, cmap='rainbow')
_ = area_eval.plot_streamline('u', 'v', cmap = 'jet')
```

### 4.2 Visualize Neural Network data

```python
_ = area_eval.plot_loss_curve(log_scale=True)
```
