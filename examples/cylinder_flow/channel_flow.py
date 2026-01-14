import deepflow as df
print("Deepflow is runned on:", df.device) # to change to cpu use df.device = 'cpu'
df.manual_seed(69) # for reproducibility

circle = df.geometry.circle(0.2, 0.2, 0.05)
rectangle = df.geometry.rectangle([0,1.1], [0,0.41])
area = rectangle - circle

domain = df.domain(rectangle.bound_list, circle.bound_list, area)
domain.show_setup()

domain.bound_list[0].define_bc({'u': ['y', lambda x:  4*1*(0.41-x)*x/0.41**2], 'v': 0})
domain.bound_list[1].define_bc({'u': 0,'v': 0})
domain.bound_list[2].define_bc({'p': 0})
domain.bound_list[3].define_bc({'u': 0,'v': 0})
domain.bound_list[4].define_bc({'u': 0, 'v': 0})
domain.bound_list[5].define_bc({'u': 0, 'v': 0})
domain.area_list[0].define_pde(df.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000))
domain.show_setup()

domain.sampling_random(bound_sampling_res=[2000, 2000, 2000, 2000, 4000, 4000])
domain.sampling_random(area_sampling_res=[10000])
domain.show_coordinates(display_conditions=False)

model0 = df.PINN(width=50, length=5)

# Define the loss calculation function
def calc_loss(model):
    bc_loss = sum(b.calc_loss(model) for b in domain.bound_list)
    pde_loss = sum(a.calc_loss(model) for a in domain.area_list)
    total_loss = bc_loss + pde_loss # weight bc_loss more

    return {"bc_loss": bc_loss, "pde_loss": pde_loss, "total_loss": total_loss} # MUST RETURN IN THIS FORMAT

# Train the model
model1, model1best = model0.train_adam(
    learning_rate=0.001,
    epochs=2000,
    calc_loss=calc_loss,
    print_every=200,
    threshold_loss=0.001)

# Train the model
model2 = model1best.train_lbfgs(
    calc_loss=calc_loss,
    epochs=500,
    print_every=50,
    threshold_loss=0.0001,)

# Create object for evaluation
area_eval = domain.area_list[0].evaluate(model2)
# Sampling uniform points
area_eval.sampling_area([300, 200])