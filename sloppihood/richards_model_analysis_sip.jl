# -----------------------------------------
# --- setup new likelihood in 'sloppihood informed' param ---
#   (assumes x,y version of prob defined)
#       x,y usual. X, Y: SIP-based
# -----------------------------------------
# # not sloppy
# XYtoxy(XY) = [XY[1]^(12/25)*XY[2]^(16/25); (XY[1]^(12/25)*XY[2]^(16/25))^(3/4)/XY[1]] 
# xytoXY(xy) = [xy[1]^(0.5)/xy[2]; xy[1]*xy[2]^0.5]
# # new variable names
# varnames["X"]="λ^(3/4)/β"
# varnames["Y"]="λβ^(3/4)"
# # sloppy
# XYtoxy(XY) = [XY[1]^(2/5)*XY[2]^(4/5); (XY[1]^(2/5)*XY[2]^(4/5))^(1/2)/XY[1]] 
# xytoXY(xy) = [xy[1]^(0.5)/xy[2]; xy[1]*xy[2]^0.5]
# # new variable names
# varnames["X"]="λ^(1/2)/β"
# varnames["Y"]="λβ^(1/2)"
# product: X = xy, Y = y/x; x = X**0.5/Y**0.5, y = X**0.5*X**0.5
XYtoxy(XY) = [(XY[1]/XY[2])^0.5; (XY[1]*XY[2])^0.5] 
xytoXY(xy) = [xy[1]*xy[2]; xy[2]/xy[1]]
# new variable names
varnames["X"]="λβ"
varnames["Y"]="β/λ"

# initial guess for optimisation
XY_initial =  xytoXY(xy_initial)# x (i.e. n) and y (i.e. p), starting guesses

# --- parameter bounds -- not monotonic, not independent. Do manual for now.
# parameter bounds
X_lb=0.05; X_ub=1.0; Y_lb=0.2; Y_ub=1.0

XY_lower_bounds = [X_lb,Y_lb]
XY_upper_bounds = [X_ub,Y_ub]

# correct initial guesses if needed
if XY_initial[1] <= XY_lower_bounds[1] || XY_initial[1] >= XY_upper_bounds[1]
    XY_initial[1] = mean([XY_lower_bounds[1],XY_upper_bounds[1]])
end
if XY_initial[2] <= XY_lower_bounds[2] || XY_initial[2] >= XY_upper_bounds[2]
    XY_initial[2] = mean([XY_lower_bounds[2],XY_upper_bounds[2]])
end

# new true value
XY_true = xytoXY(xy_true)
# new likelihood
lnlike_XY = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy)

# -----------------------------------------
# --- redo analysis with new likelihood ---
# -----------------------------------------

# --- point optimisation in original via profiling out all parameters
target_indices = [] # for point estimation whole parameter is nuisance!
lbs_all = XY_lower_bounds
ubs_all = XY_upper_bounds
nuisance_guess = XY_initial
# below 'profiles out' all parameters = does MLE
XY_value, lnlike_XY_value = SloppihoodTools.profile_target(lnlike_XY,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=[1000])
XY_MLE = XY_value
# ---

# --- full likelihood grid evaluation as targeting all parameters and profiling out none
grid_steps = [500]
target_indices = [1,2]
lbs_all = XY_lower_bounds
ubs_all = XY_upper_bounds
nuisance_guess = []
XY_values, lnlike_XY_values = SloppihoodTools.profile_target(lnlike_XY,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
max_indices = argmax(lnlike_XY_values)
XY_MLE_grid = XY_values[max_indices]

# - plot above
# split into grid components for plotting. Need unique to undo Cartesian product
X_values = unique([X for (X, _) in XY_values])
Y_values = unique([Y for (_, Y) in XY_values])
lnlike_XY_values = reshape(lnlike_XY_values,length(X_values),length(Y_values))
# convert to likelihood scale. Note: returned normalised already
like_XY_values = exp.(lnlike_XY_values)
# contour plots with chi square calibration
df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)
plt = contourf(X_values,Y_values,like_XY_values',color=:dense,levels=20,lw=0)
contour!(X_values,Y_values,like_XY_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
xlabel!(varnames["X"]) #plot!(xlabel=varnames["X"])
ylabel!(varnames["Y"])
# add best and true
max_indices = argmax(like_XY_values)
X_max = X_values[max_indices[1]]
Y_max = Y_values[max_indices[2]]
vline!([X_max],color=:black,ls=:dash,legend=false)
hline!([Y_max],color=:black,ls=:dash,legend=false)
vline!([XY_true[1]],color=:gray,lw=2,legend=false)
hline!([XY_true[2]],color=:gray,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_XY_likelihood.svg")

# --- quadratic approximation at MLE
XY_centre = XY_MLE_grid # use grid for consistent comparison with 1D
lnlike_XY_ellipse, H_XY_ellipse = SloppihoodTools.construct_ellipse_lnlike_approx(lnlike_XY,XY_centre)

# - grid evaluation as targeting all parameters and profiling out none
target_indices = [1,2]
lbs_all = XY_lower_bounds
ubs_all = XY_upper_bounds
nuisance_guess = []
XY_values, lnlike_XY_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)

# - plot above
# split into grid components for plotting. Need unique to undo Cartesian product
X_values = unique([X for (X, _) in XY_values])
Y_values = unique([Y for (_, Y) in XY_values])
lnlike_XY_ellipse_values = reshape(lnlike_XY_ellipse_values,length(X_values),length(Y_values))
# convert to likelihood scale. Note: returned normalised already
like_XY_ellipse_values = exp.(lnlike_XY_ellipse_values)
# contour plots with chi square calibration
df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)
plt = contourf(X_values,Y_values,like_XY_ellipse_values',color=:dense,levels=20,lw=0)
contour!(X_values,Y_values,like_XY_ellipse_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
xlabel!(varnames["X"])#plot!(xlabel=varnames["X"])
ylabel!(varnames["Y"])

# add best and true
X_max = XY_MLE[1]
Y_max = XY_MLE[2]
vline!([X_max],color=:black,ls=:dash,legend=false)
hline!([Y_max],color=:black,ls=:dash,legend=false)
vline!([XY_true[1]],color=:gray,lw=2,legend=false)
hline!([XY_true[2]],color=:gray,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_XY_likelihood_quadratic.svg")

# - compare ellipse and true
plt = contourf(X_values,Y_values,like_XY_values',color=:dense,levels=20,lw=0)
contour!(X_values,Y_values,like_XY_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
contour!(X_values,Y_values,like_XY_ellipse_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2,ls=:dash)
xlabel!(varnames["X"]) #plot!(xlabel=varnames["X"])
ylabel!(varnames["Y"])
# add best and true
max_indices = argmax(like_XY_values)
X_max = X_values[max_indices[1]]
Y_max = Y_values[max_indices[2]]
vline!([X_max],color=:black,ls=:dash,legend=false)
hline!([Y_max],color=:black,ls=:dash,legend=false)
vline!([XY_true[1]],color=:gray,lw=2,legend=false)
hline!([XY_true[2]],color=:gray,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_XY_likelihood_comparison.svg")

# --- 1D profiles 
# - First X (i.e. ln n) in full
target_indices = [1] #
nuisance_indices = [2]
lbs_all = XY_lower_bounds
ubs_all = XY_upper_bounds
nuisance_guess = XY_initial[nuisance_indices]
# gets interest and nuisance values along profile
XY_values, lnlike_XY_values = SloppihoodTools.profile_target(lnlike_XY,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
X_values = [X for (X, _) in XY_values] # no need for unique as 1D curve
# get location of max
max_indices = argmax(like_XY_values)
X_max = X_values[max_indices[1]]
# plot
plt = plot(X_values,exp.(lnlike_XY_values), xlabel=varnames["X"], ylabel="profile likelihood",color=:black,lw=2,legend=false)
vline!([X_max],color=:black,ls=:dash)
hline!([exp(-quantile(Chisq(1),0.95)/2)],color=:black,ls=:dash)
vline!([XY_true[1]],lw=2,color=:gray)

# - Add X (i.e. ln n) from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
XY_values, lnlike_XY_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
X_values = [X for (X, _) in XY_values] # no need for unique as 1D curve
# add to plot
plot!(X_values,exp.(lnlike_XY_ellipse_values), xlabel=varnames["X"], ylabel="profile likelihood",color=:black,ls=:dash,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_XY_likelihood_Y.svg")

# - Now Y (i.e. ln p)
target_indices = [2] #
nuisance_indices = [1]
lbs_all = XY_lower_bounds
ubs_all = XY_upper_bounds
nuisance_guess = XY_initial[nuisance_indices]
# gets interest and nuisance values along profile
XY_values, lnlike_XY_values = SloppihoodTools.profile_target(lnlike_XY,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
Y_values = [Y for (_, Y) in XY_values] # no need for unique as 1D curve
# get location of max
max_indices = argmax(like_XY_values)
Y_max = Y_values[max_indices[2]]
# plot
plt = plot(Y_values,exp.(lnlike_XY_values), xlabel=varnames["Y"], ylabel="profile likelihood",color=:black,lw=2,legend=false)
vline!([Y_max],color=:black,ls=:dash)
hline!([exp(-quantile(Chisq(1),0.95)/2)],color=:black,ls=:dash)
vline!([XY_true[2]],lw=2,color=:gray)

# - Add Y (i.e. ln p) from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
XY_values, lnlike_XY_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
Y_values = [Y for (_, Y) in XY_values] # no need for unique as 1D curve
# add to plot
plot!(Y_values,exp.(lnlike_XY_ellipse_values), xlabel=varnames["Y"], ylabel="profile likelihood",color=:black,ls=:dash,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_XY_likelihood_X.svg")