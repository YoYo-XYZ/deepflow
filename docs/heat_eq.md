# 2D Unsteady Heat Equation

This notebook demonstrates solving Unsteady 2D Fourier's Heat Equation using PINNs.

## 1. Define Geometry Domain

Set up the computational domain.

```python
import deepflow as df
print("Deepflow is runned on:", df.device) # to change to cpu use df.device = 'cpu'
df.manual_seed(69) # for reproducibility

rectangle = df.geometry.rectangle([0, 1], [0, 1])
# It seems the example used a second rectangle for domain definition in logic but passed area_list
rectangle1 = df.geometry.rectangle([0, 1], [0, 1])
domain = df.domain(rectangle, rectangle1.area_list)
domain.show_setup()
```

## 2. Define Physics

Define the Heat equation and apply boundary conditions.

```python
# Define Boundary Conditions
domain.bound_list[0].define_bc({'u': 0})   # Inflow: u=1
domain.bound_list[1].define_bc({'u': 0})   # Inflow: u=1
domain.bound_list[2].define_bc({'u': 0})   # Inflow: u=1
domain.bound_list[3].define_bc({'u': 1})  # Wall: No slip

# Define PDE (Heat Equation) and Initial Condition
domain.area_list[0].define_pde(df.pde.HeatEquation(0.1))
domain.area_list[1].define_ic({'u': 0})

# Define time domain
for g in domain:
    g.define_time(range_t = [0, 1], sampling_scheme='random')

domain.show_setup()
```

### 2. Generate Training Data

Sample initial points for training. After sampling, Deepflow will automatically generate training datasets based on the defined physics.

```python
# Sample points: [Left, Bottom, Right, Top], [Interior]
domain.sampling_lhs([1000, 1000, 2000, 1000], [2000, 2000])
domain.show_coordinates(display_physics=True)
```

## 3. Train the Model

Define the resampling scheme during training. [R3](https://arxiv.org/abs/2207.02338) scheme is recommended.

```python
def do_in_adam(epoch, model):
    if epoch % 1000 == 0 and epoch > 0:
        domain.sampling_R3([1000, 1000, 2000, 1000], [2000, 2000])
        print(domain)
        
def do_in_lbfgs(epoch, model):
    if epoch % 100 == 0 and epoch > 0:
        domain.sampling_R3([1000, 1000, 2000, 1000], [2000, 2000])
        print(domain)
```

Train the model using Adam optimizer followed by L-BFGS optimizer.

```python
model0 = df.PINN(width=32, length=4, input_vars=['x','y','t'], output_vars=['u'])
model1, model1_best = model0.train_adam(
    calc_loss = df.calc_loss_simple(domain),
    learning_rate=0.004,
    do_between_epochs=do_in_adam,
    epochs=2000)
    
model2 = model1_best.train_lbfgs(calc_loss = df.calc_loss_simple(domain), epochs=450, do_between_epochs=do_in_lbfgs, threshold_loss=5e-3)

domain.show_coordinates(display_physics=False)
```

## 4. Visualization

### 4.1 Visualize PDE area

```python
# Evaluate the best model
prediction = domain.area_list[0].evaluate(model2)
prediction.sampling_area([200, 200])
prediction.define_time(0.5)

# Plot Temperature Field
_ = prediction.plot('u', color='plasma')
_.savefig('heat_eq_u.png', dpi=200)

# Plot Training Loss
_ =prediction.plot_loss_curve(log_scale=True, keys=['total_loss'])
```

### 4.2 Animate the solution over time

```python
prediction.sampling_area([160, 160])
prediction.plot_animate('u', range_t = [0.02, 1.02], dt = 0.02, frame_interval=100, cmap='plasma', plot_type='scatter', s=1.7).save('heat_equation.mp4', dpi = 200)
```
