# treeffuser.scaler module

### *class* treeffuser.scaler.ScalerMixedTypes(scaler=None)

Bases: `object`

Data scaler for mixed data-types with continuous and categorical features.

Scale continuous features and leave categorical features unchanged. By default, the scaling
is done using StandardScaler from scikit-learn, which standardizes the data to have mean 0
and standard deviation 1.

This class does not check the input data. The indices of the categorical features must be
provided, or they will be treated as continuous and scaled.

* **Parameters:**
  **scaler** (*scaler from sklearn.preprocessing* *,* *optional*) – The scaler to use for continuous features. Default is StandardScaler if not provided.

#### fit(X: Float[ndarray, 'batch x_dim'], cat_idx: List[int] | None = None)

Fit the scaler provided at initialization to the data.

* **Parameters:**
  * **X** (*ndarray* *of* *shape* *(**batch* *,* *x_dim* *)*) – The data to fit the scaler to.
  * **cat_idx** (*list* *of* *int* *,* *optional*) – The indices of the categorical features.

#### fit_transform(X: Float[ndarray, 'batch x_dim'], cat_idx: List[int] | None = None)

Fit the scaler and transform the data in one step.

* **Parameters:**
  * **X** (*ndarray* *of* *shape* *(**batch* *,* *x_dim* *)*) – The data to fit and transform.
  * **cat_idx** (*list* *of* *int* *,* *optional*) – The indices of the categorical features.
* **Returns:**
  **X_transformed** – The transformed data with scaled continuous features.
* **Return type:**
  ndarray of shape (batch, x_dim)

#### SEE ALSO
[`fit`](#treeffuser.scaler.ScalerMixedTypes.fit)
: Fit the preprocessor to the data.

[`transform`](#treeffuser.scaler.ScalerMixedTypes.transform)
: Transform the data.

#### inverse_transform(X: Float[ndarray, 'batch x_dim'])

Takes the data back to the original scale.

* **Parameters:**
  **X** (*ndarray* *of* *shape* *(**batch* *,* *x_dim* *)*) – The data to transform back.
* **Returns:**
  **X_untransformed** – The untransformed data with the original scale.
* **Return type:**
  ndarray of shape (batch, x_dim)

#### transform(X: Float[ndarray, 'batch x_dim'])

Standardize/scale the data. The categorical features are left unchanged.

* **Parameters:**
  **X** (*ndarray* *of* *shape* *(**batch* *,* *x_dim* *)*) – The data to transform.
* **Returns:**
  **X_transformed** – The transformed data with scaled continuous features.
* **Return type:**
  ndarray of shape (batch, x_dim)
