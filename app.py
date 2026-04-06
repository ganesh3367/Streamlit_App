import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


st.set_page_config(
    page_title="Linear Regression Explorer",
    layout="wide",
)


def generate_dataset(
    kind: str,
    n_points: int,
    true_slope: float,
    true_intercept: float,
    x_span: float,
    noise_level: float,
    outlier_fraction: float,
    outlier_strength: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(-x_span, x_span, n_points)
    x = x + rng.normal(0, x_span * 0.04, size=n_points)
    x = np.sort(x)

    base_y = true_slope * x + true_intercept

    if kind == "Clean linear":
        effective_noise = 0.0
        effective_outliers = 0.0
    elif kind == "Noisy linear":
        effective_noise = max(noise_level, 1.0)
        effective_outliers = 0.0
    elif kind == "Outlier challenge":
        effective_noise = max(noise_level, 0.7)
        effective_outliers = max(outlier_fraction, 0.12)
    else:
        effective_noise = noise_level
        effective_outliers = outlier_fraction

    y = base_y + rng.normal(0, effective_noise, size=n_points)
    is_outlier = np.zeros(n_points, dtype=bool)

    n_outliers = int(round(effective_outliers * n_points))
    if n_outliers > 0:
        outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
        is_outlier[outlier_idx] = True
        jumps = rng.normal(0, outlier_strength, size=n_outliers)
        y[outlier_idx] += jumps

    return x, y, base_y, is_outlier


def predict(x, slope, intercept):
    return slope * x + intercept


def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def gradients(x, y, slope, intercept):
    preds = predict(x, slope, intercept)
    residuals = preds - y
    dm = float((2 / len(x)) * np.sum(x * residuals))
    db = float((2 / len(x)) * np.sum(residuals))
    return dm, db


def fit_closed_form(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def run_gradient_descent(x, y, start_slope, start_intercept, learning_rate, steps):
    slope = float(start_slope)
    intercept = float(start_intercept)
    history = []

    for step in range(steps + 1):
        preds = predict(x, slope, intercept)
        loss = mean_squared_error(y, preds)
        dm, db = gradients(x, y, slope, intercept)
        history.append(
            {
                "step": step,
                "slope": slope,
                "intercept": intercept,
                "loss": loss,
                "grad_slope": dm,
                "grad_intercept": db,
                "grad_norm": float(np.sqrt(dm**2 + db**2)),
            }
        )
        slope -= learning_rate * dm
        intercept -= learning_rate * db

    return pd.DataFrame(history)


@st.cache_data(show_spinner=False)
def compute_loss_surface(x, y, slope_min, slope_max, intercept_min, intercept_max, resolution):
    slope_values = np.linspace(slope_min, slope_max, resolution)
    intercept_values = np.linspace(intercept_min, intercept_max, resolution)
    loss_grid = np.zeros((len(intercept_values), len(slope_values)))

    for i, intercept in enumerate(intercept_values):
        preds = slope_values[None, :] * x[:, None] + intercept
        loss_grid[i, :] = np.mean((y[:, None] - preds) ** 2, axis=0)

    return slope_values, intercept_values, loss_grid


def add_section_copy(title, explanation, takeaway):
    st.subheader(title)
    st.write(explanation)
    st.caption(f"What to observe: {takeaway}")


def style_axis(ax):
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


st.title("Linear Regression Learning Lab")
st.write(
    "Manipulate the line, watch the error change, and trace how gradient descent learns "
    "the best-fit parameters."
)

with st.sidebar:
    st.header("Controls")
    dataset_kind = st.selectbox(
        "Dataset",
        ["Clean linear", "Noisy linear", "Outlier challenge", "Custom mix"],
    )
    seed = st.slider("Random seed", 0, 100, 7)
    n_points = st.slider("Number of points", 20, 150, 60)
    x_span = st.slider("X range", 4.0, 15.0, 8.0, 0.5)

    st.divider()
    st.subheader("Ground truth data")
    true_slope = st.slider("True slope", -4.0, 4.0, 1.8, 0.1)
    true_intercept = st.slider("True intercept", -8.0, 8.0, 1.0, 0.1)
    noise_level = st.slider("Noise level", 0.0, 8.0, 1.5, 0.1)
    outlier_fraction = st.slider("Outlier fraction", 0.0, 0.35, 0.08, 0.01)
    outlier_strength = st.slider("Outlier strength", 1.0, 25.0, 10.0, 0.5)

    st.divider()
    st.subheader("Manual line")
    manual_slope = st.slider("Manual slope (m)", -6.0, 6.0, 0.5, 0.1)
    manual_intercept = st.slider("Manual intercept (b)", -12.0, 12.0, 0.0, 0.1)
    show_true_line = st.toggle("Show true generating line", True)
    show_best_fit = st.toggle("Show optimal least-squares line", True)
    show_candidate_lines = st.toggle("Show candidate lines", False)

    st.divider()
    st.subheader("Gradient descent")
    gd_start_slope = st.slider("Start slope", -6.0, 6.0, -3.0, 0.1)
    gd_start_intercept = st.slider("Start intercept", -12.0, 12.0, 6.0, 0.1)
    learning_rate = st.slider("Learning rate", 0.001, 0.300, 0.050, 0.001)
    gd_steps = st.slider("Gradient descent steps", 5, 120, 40)
    contour_resolution = st.slider("Loss surface resolution", 25, 100, 50, 5)

x, y, base_y, is_outlier = generate_dataset(
    dataset_kind,
    n_points,
    true_slope,
    true_intercept,
    x_span,
    noise_level,
    outlier_fraction,
    outlier_strength,
    seed,
)

manual_preds = predict(x, manual_slope, manual_intercept)
manual_residuals = y - manual_preds
manual_mse = mean_squared_error(y, manual_preds)
best_slope, best_intercept = fit_closed_form(x, y)
best_preds = predict(x, best_slope, best_intercept)
best_mse = mean_squared_error(y, best_preds)
gd_history = run_gradient_descent(
    x,
    y,
    gd_start_slope,
    gd_start_intercept,
    learning_rate,
    gd_steps,
)

gd_final = gd_history.iloc[-1]
slope_pad = max(1.5, abs(best_slope - manual_slope) + 1.5, abs(best_slope - gd_start_slope) + 1.5)
intercept_pad = max(
    2.0,
    abs(best_intercept - manual_intercept) + 2.0,
    abs(best_intercept - gd_start_intercept) + 2.0,
)
slope_values, intercept_values, loss_grid = compute_loss_surface(
    x,
    y,
    best_slope - slope_pad,
    best_slope + slope_pad,
    best_intercept - intercept_pad,
    best_intercept + intercept_pad,
    contour_resolution,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Manual line MSE", f"{manual_mse:.3f}")
metric_cols[1].metric("Optimal line MSE", f"{best_mse:.3f}")
metric_cols[2].metric("Best-fit slope", f"{best_slope:.3f}")
metric_cols[3].metric("Best-fit intercept", f"{best_intercept:.3f}")

tabs = st.tabs(
    [
        "Data & Line Fit",
        "Error / Loss",
        "Loss Landscape",
        "Gradient Descent",
        "Learning Rate Experiments",
        "Noise & Outliers",
    ]
)

with tabs[0]:
    add_section_copy(
        "1. Data Distribution & Line Fitting",
        "Linear regression searches a hypothesis space of straight lines of the form y = mx + b. "
        "Use the manual sliders to move the line and judge how well it explains the scatter.",
        "A good line passes through the center of the point cloud and balances errors above and below it.",
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = np.where(is_outlier, "#d95f02", "#1f77b4")
    ax.scatter(x, y, c=colors, s=55, alpha=0.85, edgecolors="white", linewidths=0.7, label="Observed data")
    ax.plot(x, manual_preds, color="#111111", linewidth=2.5, label="Manual line")

    if show_true_line:
        ax.plot(x, base_y, color="#2ca02c", linestyle="--", linewidth=2, label="True line")
    if show_best_fit:
        ax.plot(x, best_preds, color="#d62728", linewidth=2.2, label="Optimal fit")
    if show_candidate_lines:
        candidate_offsets = [(-0.9, -1.8), (0.7, 1.4), (1.2, -1.0)]
        for i, (dm, db) in enumerate(candidate_offsets, start=1):
            ax.plot(
                x,
                predict(x, manual_slope + dm, manual_intercept + db),
                linewidth=1.1,
                alpha=0.45,
                label=f"Candidate {i}",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Observed data and competing regression lines")
    style_axis(ax)
    ax.legend(loc="best")
    st.pyplot(fig)

    info_cols = st.columns(3)
    info_cols[0].info(f"Manual line: y = {manual_slope:.2f}x + {manual_intercept:.2f}")
    info_cols[1].info(f"True line: y = {true_slope:.2f}x + {true_intercept:.2f}")
    info_cols[2].info(f"Least-squares line: y = {best_slope:.2f}x + {best_intercept:.2f}")

with tabs[1]:
    add_section_copy(
        "2. Error / Loss Function (MSE)",
        "Each vertical gap between a point and the line is a residual. Mean Squared Error squares "
        "those gaps so larger mistakes matter much more than small ones.",
        "When the line drifts away from the data, a few large residuals can dominate the total loss.",
    )
    left, right = st.columns([1.2, 1])

    with left:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.scatter(x, y, c=np.where(is_outlier, "#d95f02", "#1f77b4"), s=55, alpha=0.85)
        ax.plot(x, manual_preds, color="#111111", linewidth=2.5)
        for xi, yi, pi in zip(x, y, manual_preds):
            ax.plot([xi, xi], [yi, pi], color="#7f7f7f", alpha=0.55, linewidth=1)
        ax.set_title("Residuals relative to the current manual line")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        style_axis(ax)
        st.pyplot(fig)

    with right:
        squared_errors = manual_residuals**2
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.bar(np.arange(len(x)), squared_errors, color=np.where(is_outlier, "#d95f02", "#4c78a8"), alpha=0.85)
        ax.set_title("Squared residuals")
        ax.set_xlabel("Point index")
        ax.set_ylabel("Residual^2")
        style_axis(ax)
        st.pyplot(fig)

        st.metric("Current MSE", f"{manual_mse:.4f}")
        st.metric("Current RMSE", f"{np.sqrt(manual_mse):.4f}")
        largest_idx = int(np.argmax(squared_errors))
        st.write(
            f"Largest squared error occurs at point {largest_idx} with residual "
            f"{manual_residuals[largest_idx]:.3f}."
        )

with tabs[2]:
    add_section_copy(
        "3. Loss Surface (Parameter Space)",
        "The loss depends on both slope and intercept, so we can visualize a landscape J(m, b). "
        "The bowl-shaped valley shows where parameter combinations produce smaller error.",
        "The best-fit line sits near the bottom of the basin, and your current line appears as a point elsewhere on the surface.",
    )
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        contour = ax.contourf(slope_values, intercept_values, loss_grid, levels=25, cmap="viridis")
        ax.contour(slope_values, intercept_values, loss_grid, levels=12, colors="white", alpha=0.25, linewidths=0.7)
        ax.scatter(manual_slope, manual_intercept, color="white", edgecolors="black", s=90, label="Manual line")
        ax.scatter(best_slope, best_intercept, color="#ff4b4b", s=90, label="Optimal fit")
        gd_path = gd_history[["slope", "intercept"]].to_numpy()
        ax.plot(gd_path[:, 0], gd_path[:, 1], color="#ffdd57", linewidth=2, marker="o", markersize=3, label="GD path")
        ax.set_xlabel("Slope (m)")
        ax.set_ylabel("Intercept (b)")
        ax.set_title("Contour map of J(m, b)")
        ax.legend(loc="upper right")
        fig.colorbar(contour, ax=ax, label="MSE")
        st.pyplot(fig)

    with right:
        fig = plt.figure(figsize=(8.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        slope_mesh, intercept_mesh = np.meshgrid(slope_values, intercept_values)
        ax.plot_surface(slope_mesh, intercept_mesh, loss_grid, cmap="viridis", alpha=0.82, linewidth=0)
        ax.scatter(manual_slope, manual_intercept, manual_mse, color="black", s=45, label="Manual line")
        ax.scatter(best_slope, best_intercept, best_mse, color="#ff4b4b", s=55, label="Optimal fit")
        ax.set_xlabel("Slope")
        ax.set_ylabel("Intercept")
        ax.set_zlabel("MSE")
        ax.set_title("3D loss surface")
        ax.view_init(elev=28, azim=-56)
        ax.legend(loc="upper left")
        st.pyplot(fig)

with tabs[3]:
    add_section_copy(
        "4. Gradient Descent (Optimization Process)",
        "Gradient descent updates the parameters in the direction that most reduces loss. "
        "Each step uses the slope of the loss surface to move downhill.",
        "The path should curve toward the least-squares solution when the learning rate is reasonable.",
    )
    selected_step = st.slider("Inspect gradient descent step", 0, gd_steps, min(10, gd_steps))
    selected_row = gd_history.iloc[selected_step]
    selected_preds = predict(x, selected_row["slope"], selected_row["intercept"])

    left, right = st.columns([1.1, 1])

    with left:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.contourf(slope_values, intercept_values, loss_grid, levels=25, cmap="viridis")
        ax.plot(gd_history["slope"], gd_history["intercept"], color="#ffdd57", marker="o", markersize=3, linewidth=2)
        ax.scatter(selected_row["slope"], selected_row["intercept"], color="white", edgecolors="black", s=110)
        ax.scatter(best_slope, best_intercept, color="#ff4b4b", s=90)
        ax.set_xlabel("Slope (m)")
        ax.set_ylabel("Intercept (b)")
        ax.set_title("Optimization trajectory on the contour map")
        style_axis(ax)
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.scatter(x, y, c=np.where(is_outlier, "#d95f02", "#1f77b4"), s=55, alpha=0.85)
        ax.plot(x, selected_preds, color="#111111", linewidth=2.6, label=f"Step {selected_step}")
        ax.plot(x, best_preds, color="#d62728", linewidth=2, linestyle="--", label="Optimal fit")
        ax.set_title("Data fit at the selected optimization step")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        style_axis(ax)
        st.pyplot(fig)

    trend_cols = st.columns(4)
    trend_cols[0].metric("Step", int(selected_row["step"]))
    trend_cols[1].metric("Slope", f'{selected_row["slope"]:.4f}')
    trend_cols[2].metric("Intercept", f'{selected_row["intercept"]:.4f}')
    trend_cols[3].metric("Loss", f'{selected_row["loss"]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(gd_history["step"], gd_history["loss"], color="#2c7fb8", linewidth=2.2)
    ax.scatter(selected_step, selected_row["loss"], color="#ff4b4b", s=70)
    ax.set_title("Loss over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    style_axis(ax)
    st.pyplot(fig)

with tabs[4]:
    add_section_copy(
        "5. Learning Rate & Convergence Behavior",
        "The learning rate controls step size. Small rates converge slowly, good rates move smoothly, "
        "and large rates can overshoot the minimum or even diverge.",
        "Compare the three curves and notice how the same starting point behaves very differently under different step sizes.",
    )
    experiment_rates = [
        max(0.001, learning_rate / 5),
        learning_rate,
        min(0.6, learning_rate * 3),
    ]
    labels = ["Small rate", "Chosen rate", "Large rate"]
    colors = ["#4c78a8", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 4.8))

    summaries = []
    for rate, label, color in zip(experiment_rates, labels, colors):
        history = run_gradient_descent(x, y, gd_start_slope, gd_start_intercept, rate, gd_steps)
        ax.plot(history["step"], history["loss"], label=f"{label} ({rate:.3f})", color=color, linewidth=2)
        summaries.append((label, rate, float(history["loss"].iloc[-1])))

    ax.set_title("Loss vs iteration for different learning rates")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend(loc="best")
    style_axis(ax)
    st.pyplot(fig)

    for label, rate, final_loss in summaries:
        st.write(f"{label} at learning rate {rate:.3f} ends with loss {final_loss:.4f}.")

with tabs[5]:
    add_section_copy(
        "6. Effect of Noise & Outliers",
        "Because MSE squares residuals, a few extreme points can pull the fitted line strongly. "
        "This is why linear regression is sensitive to outliers.",
        "Compare the fit on the clean data to the fit on the noisy or outlier-rich version and watch the optimal line shift.",
    )
    clean_x, clean_y, clean_base, clean_outliers = generate_dataset(
        "Clean linear",
        n_points,
        true_slope,
        true_intercept,
        x_span,
        noise_level,
        0.0,
        outlier_strength,
        seed,
    )
    stressed_x, stressed_y, stressed_base, stressed_outliers = generate_dataset(
        "Custom mix",
        n_points,
        true_slope,
        true_intercept,
        x_span,
        max(noise_level, 2.0),
        max(outlier_fraction, 0.18),
        outlier_strength,
        seed,
    )
    clean_slope, clean_intercept = fit_closed_form(clean_x, clean_y)
    stressed_slope, stressed_intercept = fit_closed_form(stressed_x, stressed_y)

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.scatter(clean_x, clean_y, c="#1f77b4", s=55, alpha=0.85)
        ax.plot(clean_x, clean_base, color="#2ca02c", linestyle="--", linewidth=2, label="True line")
        ax.plot(clean_x, predict(clean_x, clean_slope, clean_intercept), color="#111111", linewidth=2.5, label="Fit on clean data")
        ax.set_title("Fit on clean data")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        style_axis(ax)
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.scatter(
            stressed_x,
            stressed_y,
            c=np.where(stressed_outliers, "#d95f02", "#1f77b4"),
            s=55,
            alpha=0.85,
        )
        ax.plot(stressed_x, stressed_base, color="#2ca02c", linestyle="--", linewidth=2, label="True line")
        ax.plot(
            stressed_x,
            predict(stressed_x, stressed_slope, stressed_intercept),
            color="#111111",
            linewidth=2.5,
            label="Fit with noise/outliers",
        )
        ax.set_title("Fit under noise and outliers")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        style_axis(ax)
        st.pyplot(fig)

    shift_cols = st.columns(3)
    shift_cols[0].metric("Clean fit slope", f"{clean_slope:.3f}")
    shift_cols[1].metric("Stressed fit slope", f"{stressed_slope:.3f}")
    shift_cols[2].metric("Slope shift", f"{stressed_slope - clean_slope:+.3f}")

st.divider()
st.markdown(
    """
    **Key idea:** Linear regression learns by proposing a line, measuring its error with MSE,
    and adjusting slope and intercept to move downhill on the loss surface.
    """
)
