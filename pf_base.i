[Mesh]
  type = GeneratedMesh
  dim = 2
  nx =
  ny =
  xmin =
  xmax =
  ymin =
  ymax =
  elem_type = QUAD4
[]

[Variables]

  [./c]
    [./InitialCondition]
      type = FunctionIC
      function = c_txt
    [../]
  [../]
[]

[Kernels]

  # Order parameter c
  [./dcdt]
    type = TimeDerivative
    variable = c
  [../]

  [./c_diffusion]
    type = ACInterface
    kappa_name = kc
    mob_name = L_c
    variable = c
  [../]
[]

[Materials]
  [./consts]
    type = GenericConstantMaterial
    prop_names  = 'L kappa_eta L_c'
    prop_values =
  [../]
  [./kcmap]
    type = GenericFunctionMaterial
    block = 0
    prop_names = kc
    prop_values = kc_txt
    outputs = exodus
  [../]
  [./asmap]
    type = GenericFunctionMaterial
    block = 0
    prop_names = as
    prop_values = as_txt
    outputs = exodus
  [../]
  [./free_energy_etai]
    type = DerivativeParsedMaterial
    block = 0
    property_name = F
    coupled_variables = 
    constant_names = 'h'
    constant_expressions = 
    expression = 
    enable_jit = true
    derivative_order = 2
    # outputs = exodus
  [../]
  [./Ed]
  type = DerivativeParsedMaterial
    block = 0
    property_name = Ed
    coupled_variables = 
    material_property_names = 'as'
    constant_names = 'c_eq k_diss k_prec'
    constant_expressions =
    expression = 
    enable_jit = true
    derivative_order = 2
    # outputs = exodus
  [../]
  [./free_energy_and_ed]
    type = DerivativeParsedMaterial
    block = 0
    property_name = F_total
    coupled_variables = 
    material_property_names = 
    expression = 'F+Ed'
    enable_jit = true
    derivative_order = 2
    # outputs = exodus
  [../]
[]

[Functions]

  [c_txt]
    type = PiecewiseMultilinear
    data_file = data/c.txt
  []
	[as_txt]
		type = PiecewiseMultilinear
		data_file = data/as.txt
	[]
	[kc_txt]
		type = PiecewiseMultilinear
		data_file = data/kc.txt
	[]
[]

[Preconditioning]
  # This preconditioner makes sure the Jacobian Matrix is fully populated. Our
  # kernels compute all Jacobian matrix entries.
  # This allows us to use the Newton solver below.
  [./SMP]
    type = SMP
    full = true
  [../]
[]

[Executioner]
  type = Transient
  scheme = 'bdf2'

  # Automatic differentiation provides a _full_ Jacobian in this example
  # so we can safely use NEWTON for a fast solve
  solve_type = 'NEWTON'

  l_max_its = 20
  l_tol = 
  l_abs_tol = 
  nl_max_its = 10
  nl_rel_tol = 
  nl_abs_tol = 

  start_time = 0.0
  end_time   =

  [./TimeStepper]
    type = SolutionTimeAdaptiveDT
    dt =
  [../]
[]

[Outputs]
  execute_on = 'initial timestep_end'
  exodus = true
  [./other]
    type = VTK
    execute_on = 'initial final'
  [../]
  [console]
    type = Console
    execute_on = 'nonlinear'
    all_variable_norms = true
  []
[]
