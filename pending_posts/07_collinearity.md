# Entry 7 - Collinearity

In <font color='red'>Entry 6</font> I looked at the correlation between the prediction features and the feature of interest (well, kinda. This problem question is a little weird in that I'll be predicting mass_1024kg, but am interested in how the other features relate to atmospheric_mass_kg). Now I'll look at how all of the features relate to each other.

The notebook with the code that accompanies the concepts discussed below can be found in the <font color='red'>Entry 7 notebook</font>.

## The Problem

[This Data Science Central blogger](https://www.datasciencecentral.com/profiles/blogs/multicollinearity-a-problem-or-an-opportunity) put it very sucintly: 'Collinearity is infamously famous for inflating the variance of at least one estimated regression coefficient, which can cause the model to predict erroneously.' Coefficients are also how models are intrepreted. Screwey coefficients can mess up the model's predictions and hinder understanding how the model got its results.

- [Collinearity](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/) occurs when two variables are corelated; a change in one causes changes in the other.
- [Multicollinearity](https://etav.github.io/python/vif_factor_python.html) occurs when three or more variables are corelated. Multicollinearity presents a larger challenge because it can be present even when isolated pairs of variables are not collinear.

Most models and statistical methods assume variables are independent. While having collinear features isn't always a problem, as discussed above it can cause issues in the coefficients, effecting predictions and making interpretability more difficult.

Not every problem requires interpretability, but every problem shoots for the most accurate result. [This answer](https://stats.stackexchange.com/questions/168622/why-is-multicollinearity-not-checked-in-modern-statistics-machine-learning) on stats.stackexchange makes the argument that cross-validation (more on cross-validation in a later entry) negates the risk of a slight change in the data causing huge fluctuations in coefficients (the fluctuation causing less accurate predictions).

Effects on predictions aside, the problems I address at work require interpretability. As such, in my case it's better to address collinearity than not.

## The Options

[Sabber](https://medium.com/@sabber) and I originally just looked at the correlation matrix of the features. However, having read more about multicollinearity, a method called VIF (variance inflation factor) is recommended.

The correlation matrix is basically what I did in <font color='red'>Entry 6</font>, but instead of limiting the relationship to just one target variable, I look at every how every variable is related to every other variable.

[VIF](https://www.statisticshowto.datasciencecentral.com/variance-inflation-factor/) uses the [r squared](https://www.statisticshowto.datasciencecentral.com/adjusted-r2/) value to give a kind of scaled result that can be compared across disparate values and ranges. It starts at 1 and goes up from there with no maximum limit.
- 1 = not correlated
- Between 1 and 5 = moderately correlated
- Greater than 5 = highly correlated

The general recommendation is to not worry about multicollinearity unless the VIF is larger than 5.

## The Proposed Solution

#### Correlation

The .corr() method returns each variable's correlation with every other variable in a nice correlation matrix. While I'm sure looking at a table full of numbers is what every analyst's dreams (it's not, unless you include nightmares...), it's easier/faster to see it as a visualization. The correlation matrix is easily visualized in a heatmap.

A quick sort of the correlations grouped the highly correlated feratures together for easier interpretation. The features of concern are mass_1024kg, diameter_km, mean_radius_km, equatorial_radius_km, and escape_vel_km_s. These result makes sense.

Since I'll be varying mass, I'm keeping the mass_1024kg feature. Of the other four, I intend to keep diameter. Radius is just half the diameter, so those two features are expendable. The [formula](https://en.wikibooks.org/wiki/LaTeX/Mathematics) for [escape velocity](https://www.toppr.com/guides/physics-formulas/escape-velocity-formula/) is:

$$\sqrt{\frac{2GM}{R}}$$

where G is the gravational constant, M is the mass, and R is the radius. Since it can be derived from mass and diameter, that feature gets dropped too.

The list of features to drop is mean_radius_km, equatorial_radius_km, and escape_vel_km_s.

#### Automating the Solution

Sabber came up with a way to automatically drop correlated features. See the supplementary notebook for the function along with an explanation of how it works.

The function keeps the first of a pair of correlated features. This means that the order of the features matters. A different set of features will result if the order of rows (reminder: the row names and column names are the same in the correlation matrix) are in a different order from one run to another.

For example, I ran the corr() function on the dataframe as it was loaded (after removing the irrevalent features). But I also ran a version where I sorted the features by the absolute value of the Spearman correlation (originally done so I could group related features). In one version the diameter feature was kept and the two radius features dropped. In the other version, one of the radius features was the one that was kept.

The automated method also found more features to remove than I did. It included nbr_moons and gravity_m_s2. Gravity was selected for removal because of it's correlations with mass_1024kg and escape_vel_km_s, both of which were also removed (due to their correlation with diameter). This could be a removal of multicollinear features - if there's a linear relationship between mass and gravity, and a linear relationship between gravity and diameter then there's some kind of relationship between mass and diameter (if a=b and b=c then a=c).

When it comes to this small dataset with 17 features, I feel like I want control over which of a pair of features is removed. However, when it comes to larger datasets, like the nearly 600 feature one at work, I think I prefer the automated method.

#### VIF

The use of linear correlation (usually the Pearson method) is a well documented in machine learning. The other method, VIF isn't as popular. I expected it to find additional interdependencies that correlation didn't, but it didn't return what I expected.

#### Final Features

After taking into consideration all the correlations (manual and automated recommendations) the full list of features is:

atmospheric_mass_kg, diameter_km, mass_1024kg, V(1,0) (mag), day_len_hr, surface_pressure_bars, rings, geometric_albedo, density_kg_m3, magnetic_field, orbital_inclination_degrees, type, rotation_period_hr

## The Fail

### VIF

The first problem was in the low end values: some of them were less than 1. Based on the documentation, I expected 1 to be the lowest possible result.

Secondly, VIF didn't find the direct correlations; diameter_km, mean_radius_km, and equatorial_radius_km were all around 2.5. As far as interdependence, escape velocity is literally mathematically derived from mass and radius. That feature should be above 5 at the very least, but was only 2.4.

My least favorite thing about VIF is that is doesn't indicate which variables are in a multicollinear relationship; mass_1024kg has a huge VIF, but no indicator as to which other variables it's collinearily connected to.

Based on these results, the correlation matrix catches the necessary relationships and is more explainable than the VIF. VIF is also not very interpretable. Due to these factors, I will use the correlation matrix and not VIF going forward.

## Next Up

Scale and center the values.


```python

```