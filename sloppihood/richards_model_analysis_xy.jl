# --- point optimisation in original via profiling out all parameters
target_indices = [] # for point estimation whole parameter is nuisance!
lbs_all = xy_lower_bounds
ubs_all = xy_upper_bounds
nuisance_guess = xy_initial
# below 'profiles out' all parameters = does MLE
xy_value, lnlike_xy_value = SloppihoodTools.profile_target(lnlike_xy,target_indices,lbs_all,ubs_all,nuisance_guess)
xy_MLE = xy_value
# ---

# --- full likelihood grid evaluation as targeting all parameters and profiling out none
grid_steps=[500]
target_indices = [1,2]
lbs_all = xy_lower_bounds
ubs_all = xy_upper_bounds
nuisance_guess = []
xy_values, lnlike_xy_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
max_indices = argmax(lnlike_xy_values)
xy_MLE_grid = xy_values[max_indices] # this useful for comparing ellipse approx?

# - plot above
# split into grid components for plotting. Need unique to undo Cartesian product
x_values = unique([x for (x, _) in xy_values])
y_values = unique([y for (_, y) in xy_values])
lnlike_xy_values = reshape(lnlike_xy_values,length(x_values),length(y_values))
# convert to likelihood scale. Note: returned normalised already
like_xy_values = exp.(lnlike_xy_values)
# contour plots with chi square calibration
df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)
plt = contourf(x_values,y_values,like_xy_values',color=:dense,levels=20,lw=0)
contour!(x_values,y_values,like_xy_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
xlabel!(varnames["x"]) #plot!(xlabel=varnames["x"])
ylabel!(varnames["y"])
# add best and true
max_indices = argmax(like_xy_values)
x_max = x_values[max_indices[1]]
y_max = y_values[max_indices[2]]
vline!([x_max],color=:black,ls=:dot,lw=2,legend=false)
hline!([y_max],color=:black,ls=:dot,lw=2,legend=false)
vline!([xy_true[1]],color=:gray,lw=2,legend=false)
hline!([xy_true[2]],color=:gray,lw=2,legend=false)
savefig(plt,"./richards_likelihood.svg")
display(plt)

# --- quadratic approximation at MLE
# use either point optimisation or grid opt
xy_centre = xy_MLE_grid #xy_MLE
lnlike_xy_ellipse, H_xy_ellipse = SloppihoodTools.construct_ellipse_lnlike_approx(lnlike_xy,xy_centre)

# - grid evaluation as targeting all parameters and profiling out none
target_indices = [1,2]
lbs_all = xy_lower_bounds
ubs_all = xy_upper_bounds
nuisance_guess = []
xy_values, lnlike_xy_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)

# - plot above
# split into grid components for plotting. Need unique to undo Cartesian product
x_values = unique([x for (x, _) in xy_values])
y_values = unique([y for (_, y) in xy_values])
lnlike_xy_ellipse_values = reshape(lnlike_xy_ellipse_values,length(x_values),length(y_values))
# convert to likelihood scale. Note: returned normalised already
like_xy_ellipse_values = exp.(lnlike_xy_ellipse_values)
# contour plots with chi square calibration
df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)
plt = contourf(x_values,y_values,like_xy_ellipse_values',color=:dense,levels=20,lw=0)
contour!(x_values,y_values,like_xy_ellipse_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
xlabel!(varnames["x"])#plot!(xlabel=varnames["x"])
ylabel!(varnames["y"])

# add best and true
x_max = xy_MLE[1]
y_max = xy_MLE[2]
vline!([x_max],color=:black,ls=:dot,lw=2,legend=false)
hline!([y_max],color=:black,ls=:dot,lw=2,legend=false)
vline!([xy_true[1]],color=:gray,lw=2,legend=false)
hline!([xy_true[2]],color=:gray,lw=2,legend=false)
savefig(plt,"./richards_likelihood_quadratic.svg")
display(plt)

# - compare ellipse and true
plt = contourf(x_values,y_values,like_xy_values',color=:dense,levels=20,lw=0)
contour!(x_values,y_values,like_xy_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,lw=2)
contour!(x_values,y_values,like_xy_ellipse_values',levels=[llstar95,llstar50,llstar05],color=:black,colorbar=false,ls=:dash,lw=2)
xlabel!(varnames["x"]) #plot!(xlabel=varnames["x"])
ylabel!(varnames["y"])
# add best and true
max_indices = argmax(like_xy_values)
x_max = x_values[max_indices[1]]
y_max = y_values[max_indices[2]]
vline!([x_max],color=:black,ls=:dot,lw=2,legend=false)
hline!([y_max],color=:black,ls=:dot,lw=2,legend=false)
vline!([xy_true[1]],color=:gray,lw=2,legend=false)
hline!([xy_true[2]],color=:gray,lw=2,legend=false)
savefig(plt,"./richards_likelihood_comparison.svg")
#savefig(plt,"richards-ellipse-true.svg")
display(plt)

# --- 1D profiles 
# - First x (i.e. n) in full
target_indices = [1] #
nuisance_indices = [2]
lbs_all = xy_lower_bounds
ubs_all = xy_upper_bounds
nuisance_guess = xy_initial[nuisance_indices]
# gets interest and nuisance values along profile
xy_values, lnlike_xy_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
x_values = [x for (x, _) in xy_values] # no need for unique as 1D curve
# get location of max
max_indices = argmax(like_xy_values)
x_max = x_values[max_indices[1]]
# plot
plt = plot(x_values,exp.(lnlike_xy_values), xlabel=varnames["x"], ylabel="profile likelihood",color=:black,lw=2,legend=false)
vline!([x_max],color=:black,ls=:dot,lw=2)
hline!([exp(-quantile(Chisq(1),0.95)/2)],color=:black,ls=:dot,lw=2)
vline!([xy_true[1]],color=:gray,lw=2)

# - Add x (i.e. n) from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
xy_values, lnlike_xy_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
x_values = [x for (x, _) in xy_values] # no need for unique as 1D curve
# add to plot
plot!(x_values,exp.(lnlike_xy_ellipse_values), xlabel=varnames["x"], ylabel="profile likelihood",color=:black,ls=:dash,lw=2,legend=false)
savefig(plt,"./richards_likelihood_lambda.svg")
display(plt)

# - Now y (i.e. p)
target_indices = [2] #
nuisance_indices = [1]
lbs_all = xy_lower_bounds
ubs_all = xy_upper_bounds
nuisance_guess = xy_initial[nuisance_indices]
# gets interest and nuisance values along profile
xy_values, lnlike_xy_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
y_values = [y for (_, y) in xy_values] # no need for unique as 1D curve
# get location of max
max_indices = argmax(like_xy_values)
y_max = y_values[max_indices[2]]
# plot
plt = plot(y_values,exp.(lnlike_xy_values), xlabel=varnames["y"], ylabel="profile likelihood",color=:black,lw=2,legend=false)
vline!([y_max],color=:black,ls=:dot,lw=2)
hline!([exp(-quantile(Chisq(1),0.95)/2)],color=:black,ls=:dot,lw=2)
vline!([xy_true[2]],color=:gray,lw=2)

# - Add y (i.e. n) from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
xy_values, lnlike_xy_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter
y_values = [y for (_, y) in xy_values] # no need for unique as 1D curve
# add to plot
plot!(y_values,exp.(lnlike_xy_ellipse_values), xlabel=varnames["y"], ylabel="profile likelihood",color=:black,ls=:dash,lw=2,legend=false)
display(plt)
savefig(plt,"./richards_likelihood_beta.svg")