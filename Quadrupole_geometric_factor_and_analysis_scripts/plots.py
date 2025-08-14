import numpy as np
import plotly.graph_objects as go

# Time array
t = np.linspace(0, 6 * np.pi, 1000)

# Function to calculate data
def calculate_data(E_a, E_b, delta, theta_k, omega, v):
    n_k = np.array([np.sin(theta_k), 0, np.cos(theta_k)])
    n_k /= np.linalg.norm(n_k)

    n_perp1 = np.array([-n_k[2], 0, n_k[0]]) / np.linalg.norm([-n_k[2], 0, n_k[0]])
    n_perp2 = np.array([0, 1, 0])

    s = v * t
    x, y, z = s * n_k[0], s * n_k[1], s * n_k[2]

    E_x = E_a * np.cos(omega * t + delta) * n_perp1[0] + E_b * np.sin(omega * t) * n_perp2[0]
    E_y = E_a * np.cos(omega * t + delta) * n_perp1[1] + E_b * np.sin(omega * t) * n_perp2[1]
    E_z = E_a * np.cos(omega * t + delta) * n_perp1[2] + E_b * np.sin(omega * t) * n_perp2[2]

    X = x + E_x
    Y = y + E_y
    Z = z + E_z

    return X, Y, Z

# Initial parameters
E_a = 5
E_b = 3
delta = 0
theta_k = np.pi / 6
omega = 1.0
v = 1.0

# Generate the initial data
X, Y, Z = calculate_data(E_a, E_b, delta, theta_k, omega, v)

# Create figure
fig = go.Figure(
    data=[go.Scatter3d(x=X, y=Y, z=Z, mode='lines', line=dict(color='blue', width=4))],
    layout=go.Layout(
        title="3D Polarization Helix with Multi-Parameter Sliders",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=900,
        height=700,
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="Reset",
                         method="update",
                         args=[{"x": [X], "y": [Y], "z": [Z]}])
                ]),
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                type="buttons",
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
)

# Add sliders for each parameter
sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "E_a: ", "font": {"size": 14}},
        pad={"t": 50},
        steps=[dict(
            label=str(val),
            method="update",
            args=[{"x": [calculate_data(val, E_b, delta, theta_k, omega, v)[0]],
                   "y": [calculate_data(val, E_b, delta, theta_k, omega, v)[1]],
                   "z": [calculate_data(val, E_b, delta, theta_k, omega, v)[2]]}]
        ) for val in np.linspace(1, 10, 10)]
    ),
    dict(
        active=0,
        currentvalue={"prefix": "E_b: ", "font": {"size": 14}},
        pad={"t": 10},
        steps=[dict(
            label=str(val),
            method="update",
            args=[{"x": [calculate_data(E_a, val, delta, theta_k, omega, v)[0]],
                   "y": [calculate_data(E_a, val, delta, theta_k, omega, v)[1]],
                   "z": [calculate_data(E_a, val, delta, theta_k, omega, v)[2]]}]
        ) for val in np.linspace(1, 10, 10)]
    ),
    dict(
        active=0,
        currentvalue={"prefix": "Delta: ", "font": {"size": 14}},
        pad={"t": 10},
        steps=[dict(
            label=f"{val:.2f}",
            method="update",
            args=[{"x": [calculate_data(E_a, E_b, val, theta_k, omega, v)[0]],
                   "y": [calculate_data(E_a, E_b, val, theta_k, omega, v)[1]],
                   "z": [calculate_data(E_a, E_b, val, theta_k, omega, v)[2]]}]
        ) for val in np.linspace(0, 2*np.pi, 20)]
    )
]

fig.update_layout(sliders=sliders)

# Show the figure
fig.show()
