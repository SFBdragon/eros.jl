#= The MIT License (MIT)

Copyright © 2023 Shaun Beautement

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE. =#

# This program computes simple in-plane transfer orbits between the orbits
# of two solar, in-plane, celestial objects orbitting the sun

# run with `julia eros.jl`

# to specify custom orbits or bodies, create Orbit and Body constants 
# for which examples for Earth, Eros, and Mars are included, and modify
# the beginning of the main() method at the bottom to use the correct orbit and body

import Pkg
Pkg.add("Plots")
using Plots

# ------------------ TYPES ------------------ #

struct Orbit
    a  :: Float64 # semi-major axis
    e  :: Float64 # eccentricity
    n  :: Float64 # mean motion
    ϖ  :: Float64 # longitude of periapsis
    L0 :: Float64 # mean longitude at epoch
    τ  :: Float64 # Julian Day of epoch
end


struct Body
    ω :: Float64 # rotational frequency
    μ :: Float64 # standard gravitation
    name :: String 
end

struct Coord
    R :: Float64 # distance from the origin (the sun)
    f :: Float64 # angle from the J2000 ecliptic
end

struct Transfer
    t_l       :: Float64
    t_a       :: Float64
    orbit     :: Orbit 
    Δv        :: Float64 
    leave_Δv  :: Float64
    arrive_Δv :: Float64
end

# broadcast as scalar
Base.Broadcast.broadcastable(o::Orbit   ) = Ref(o)
Base.Broadcast.broadcastable(b::Body    ) = Ref(b)
Base.Broadcast.broadcastable(c::Coord   ) = Ref(c)
Base.Broadcast.broadcastable(t::Transfer) = Ref(t)


# ------------------ CONSTANTS ------------------ #

const G          :: Float64 = 6.67430e-11
const AU         :: Float64 = 1.495978707e11
const J2000      :: Float64 = 2451545.0
const solar_mass :: Float64 = 1.98847e30

# all values in days / JD
const soonest_launch :: Float64 = 2451908.5 # 30 Dec 2000
const launch_lookahead  :: Float64 = 2000
const shortest_transfer :: Float64 = 100
const longest_transfer  :: Float64 = 500

# https://ssd.jpl.nasa.gov/planets/approx_pos.html
const earth_orbit = Orbit(
    1.00000018*AU, 
    0.01673163, 
    2π/365.256363004,
    deg2rad(102.93768193),
    deg2rad(100.46457166),
    J2000,
)

const earth = Body(
    2π/(0.99726968*24*60^2),
    3.986004418e14,
    "Earth",
)

# https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=433
const eros_orbit = Orbit(
    1.458117412303767*AU, 
    0.2227966940876033, 
    deg2rad(0.5597773600412238),
    deg2rad(123.1937561), # ϖ = Ω + ω
    deg2rad(345.9566508), # L0 = M + ϖ
    2460200.5
)

const eros = Body(
    2π/(5.270*60^2),
    4.463e5,
    "Eros",
)

# https://ssd.jpl.nasa.gov/planets/approx_pos.html
const mars_orbit = Orbit(
    1.52371034*AU, 
    0.09339410, 
    2π/686.980,
    deg2rad(336.0563704),
    deg2rad(355.446568),
    J2000,
)

const mars = Body(
    2π/(24.6597*60^2),
    4.2828e13,
    "Mars",
)


# ------------------ FUNCTIONS ------------------ #

# date to julian day
function date_to_jd(year::Integer, month::Integer, day::Integer)
    # L. E. Doggett, Ch. 12, "Calendars", p. 604, in Seidelmann 1992
    # https://en.wikipedia.org/wiki/Julian_day#Converting_Gregorian_calendar_date_to_Julian_Day_Number
    return (1461 * (year + 4800 + (month - 14)÷12))÷4 +
        (367 * (month - 2 - 12 * ((month - 14)÷12)))÷12 - 
        (3 * ((year + 4900 + (month - 14)÷12)÷100))÷4 + day - 32075
end

# julian day to date
function jd_to_date(jd::Number)
    jd = convert(Int64, floor(jd))
    # Adapted from Richards 1998, 316
    f = jd + 1401 + (((4 * jd + 274277) ÷ 146097) * 3) ÷ 4 - 38
    e = 4 * f + 3
    g = mod(e, 1461) ÷ 4
    h = 5 * g + 2
    D = mod(h, 153) ÷ 5 + 1
    M = mod(h ÷ 153 + 2, 12) + 1
    Y = e ÷ 1461 - 4716 + (12 + 2 - M) ÷ 12

    return Y, M, D
end

# time in JD to mean anomaly
function t_to_M(t::Number, o::Orbit)
    # https://farside.ph.utexas.edu/teaching/celestial/Celestial/node34.html
    return o.L0 + o.n * (t - o.τ) - o.ϖ
end

# mean anomaly to eccentric anomaly
function M_to_E(M::Float64, o::Orbit)
    # ensure M ∈ [-π, π] for quick convergence
    M = mod(M + π, 2π) - π

    # function to find degree of error, and tolerance
    error = E -> E + o.e * sin(E) - M
    ϵ = 1e-5

    # newtonian iteration 
    
    E = M # E = M is a decent guess

    while abs(error(E)) > ϵ
        E -= (E + o.e * sin(E) - M) / (1 + o.e * cos(E))
    end

    return E
end

# eccentric anomaly to mean anomaly
function E_to_M(E::Float64, e::Float64)
    return E + e * sin(E)
end

# eccentric anomaly to radius
function E_to_R(E::Float64, o::Orbit)
    return o.a * (1 - o.e * cos(E))
end

# eccentric anomaly to true anomaly
function E_to_f(E::Float64, o::Orbit)
    β = (1 + sqrt(1 - o.e^2)) / o.e
    return E + 2 * atan(sin(E)/(β-cos(E)))
end

# true anomaly to eccentric anomaly
function f_to_E(f::Float64, e::Float64)
    return 2 * atan( sqrt((1-e)/(1+e)) * tan(f/2) )
end

# true anomaly to radius
function f_to_R(f::Float64, o::Orbit)
    # rearragement of the polar equation
    return o.a * (1 - o.e^2) / (1 + o.e * cos(f))
end

# time in JD to polar position 
function t_to_r(t::Float64, o::Orbit)
    M = t_to_M(t, o)
    E = M_to_E(M, o)
    f = E_to_f(E, o)
    R = f_to_R(f, o)
    
    # rotate to J2000 ecliptic reference
    return Coord(R, f + o.ϖ)
end

# time to velocity 
function t_to_v(t::Float64, o::Orbit)
    M = t_to_M(t, o);
    E = M_to_E(M, o)
    R = E_to_R(E, o)

    # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    v = √(G * solar_mass * o.a) / R .* [-sin(E); √(1-o.e^2) * cos(E)]
    
    # rotate to J2000 ecliptic reference
    return [cos(o.ϖ) -sin(o.ϖ); sin(o.ϖ) cos(o.ϖ)] * v
end

function low_energy_apsis(f1::Float64, f2::Float64)
    return (f1 + f2 + π) * 0.5
end

function geo_orbit_v(obj::Body)
    return cbrt(obj.μ * obj.ω)
end

# ------------------ ALGORITHM ------------------ #

function transfer_orbit(L::Coord, A::Coord, t_l::Float64, h::Float64) :: Union{Orbit, Nothing}
    ϖ = low_energy_apsis(L.f, A.f) + h

    cosL = cos(L.f - ϖ)
    cosA = cos(A.f - ϖ)

    # compute the eccentricity
    denom = (L.R * cosL - A.R * cosA)
    e = (A.R - L.R) / denom

    # in order to keep e positive, negate, then flip periapsis/apoapsis
    # this occurs where the periapsis angle, ϖ needs to be the apoapsis instead
    if e < 0
        e = -e
        ϖ += π
    end

    # reject hyperbolic orbits
    if abs(e) >= 1
        return nothing
    end

    # compute semi-latus rectum, semi-major axis, and mean motion
    l = A.R * L.R * (cosL - cosA) / denom
    a = l/(1-e^2)
    n = sqrt(G * solar_mass / a^3) * (24*60^2)

    # compute the mean longitude at launch, t_l
    E_l = f_to_E(L.f - ϖ, e)
    L_l = E_to_M(E_l, e) + ϖ

    return Orbit(a, e, n, ϖ, L_l, t_l)
end

function transfer_Δvs(
    v_l_i::Vector{Float64}, 
    v_l_f::Vector{Float64}, 
    v_a_i::Vector{Float64},
    v_a_f::Vector{Float64}, 
    b_l::Body, b_a::Body
)
    # reciprocal vector lengths 
    v_l_f_rlen = 1/√sum(v_l_f.^2)
    v_a_i_rlen = 1/√sum(v_a_i.^2)

    # compute the additional relative velocity due to geostationary orbit
    rotatary_v_l = v_l_f * v_l_f_rlen * geo_orbit_v(b_l)
    rotatary_v_a = v_a_i * v_a_i_rlen * geo_orbit_v(b_a)

    # calculate candidate leaving and arrival deltavees
    Δv_l = √sum((v_l_f - v_l_i - rotatary_v_l).^2)
    # depending on which side of the planet the spaceship approaches
    # relative to its direction of rotation
    Δv_a_1 = √sum((v_a_f - v_a_i - rotatary_v_a).^2)
    Δv_a_2 = √sum((v_a_f - v_a_i + rotatary_v_a).^2)
    
    # enter orbit in whichever direction is easiest
    return Δv_l, min(Δv_a_1, Δv_a_2)
end

function find_transfers(
    soonest_launch::Float64, 
    latest_launch::Float64, 
    orbit_l::Orbit, 
    orbit_a::Orbit, 
    body_l::Body, 
    body_a::Body
)
    transfers = Transfer[]
    
    # foreach launch date
    for t_l ∈ soonest_launch:latest_launch
        # foreach arrival date
        for t_a ∈ t_l+shortest_transfer:t_l+longest_transfer
    
            r_l = t_to_r(t_l, orbit_l) 
            r_a = t_to_r(t_a, orbit_a)
    
            # for a few small variations of the transfer's argument of periapsis
            for h ∈ -0.1:0.02:0.1

                tf_orbit = transfer_orbit(r_l, r_a, t_l, h)
                
                # if the transfer is not hyperbolic (here, nothing)
                # and the eccentricity isn't far too high to be reasonable
                if !isnothing(tf_orbit) && 0 < tf_orbit.e < 0.7

                    # check whether the spaceship and celestial body coincide at t_a
                    r_a_tf = t_to_r(t_a, tf_orbit)
                    if abs(mod(r_a_tf.f - r_a.f + π, 2π) - π) > 0.01
                        continue
                    end
    
                    # calculate orbit velocities
                    v_l = t_to_v(t_l, orbit_l) 
                    v_a = t_to_v(t_a, orbit_a)
                    v_l_tf = t_to_v(t_l, tf_orbit)
                    v_a_tf = t_to_v(t_a, tf_orbit)

                    # calculate Δv
                    leave_Δv, arrive_Δv = transfer_Δvs(v_l, v_l_tf, v_a_tf, v_a, body_l, body_a)
        
                    # push the transfer data into the array
                    push!(
                        transfers, 
                        Transfer(t_l, t_a, tf_orbit, leave_Δv + arrive_Δv, leave_Δv, arrive_Δv)
                    )
                end
            end 
        end
    end

    return transfers
end

# ------------------ SCRIPT ------------------ #

function best_transfers(transfers::Array{Transfer})
    daily_best = Transfer[]

    day_ind = firstindex(transfers)
    i = firstindex(transfers) + 1
    while i <= lastindex(transfers) + 1
        if i > lastindex(transfers) || transfers[i-1].t_l + 3 < transfers[i].t_l
            day_group = day_ind:(i-1)
            day_ind = i

            day_best = argmin(t->t.Δv, transfers[day_group])
            push!(daily_best, day_best)
        end

        i += 1
    end

    good_Δv = Inf
    best_transfers = Transfer[]
    for i ∈ eachindex(daily_best)
        tf = daily_best[i]
        if tf.Δv < good_Δv
            push!(best_transfers, tf)
            good_Δv = min(good_Δv, 2 * tf.Δv)
        end
    end

    return best_transfers
end

function display_and_choose_transfers(transfers, orbit_1, orbit_2, body_1, body_2)
    orbit_num = 1

    for tf ∈ transfers
        jds = tf.t_l:tf.t_a
    
        coords_1  = t_to_r.(jds, orbit_1)
        coords_2  = t_to_r.(jds, orbit_2)
        coords_tf = t_to_r.(jds, tf.orbit)
    
        display(plot(
            [(c->c.f).(coords_1) (c->c.f).(coords_2) (c->c.f).(coords_tf)], 
            [(c->c.R).(coords_1) (c->c.R).(coords_2) (c->c.R).(coords_tf)], 
            proj=:polar, 
            linewidth=range(1,10,length=length(jds)), 
            marker=2,
            lab=[body_1.name body_2.name "Transfer"],
            title=string("Transfer Orbit - Launch: ", jd_to_date(tf.t_l)),
        ))
    
        println("ORBIT NUMBER: ", orbit_num)
        println("Launch JD: ", tf.t_l, " Y/M/D: ", jd_to_date(tf.t_l))
        println("Arrival JD: ", tf.t_a, " Y/M/D: ", jd_to_date(tf.t_a))
        println("Δv: ", tf.Δv)
        println("Transfer Orbit: ", tf.orbit)
    
        readline()

        orbit_num += 1
    end
    
    print("Choose an orbit to take (1-", orbit_num-1, "): ")
    return parse(Int64, readline())
end

function main()
    orbit_1 = earth_orbit
    body_1 = earth
    orbit_2 = eros_orbit
    body_2 = eros
    
    transfers = find_transfers(
        soonest_launch, 
        soonest_launch + launch_lookahead, 
        orbit_1, 
        orbit_2, 
        body_1, 
        body_2,
    )
    
    # plot energies
    display(plot(
        (t->t.t_l).(transfers), 
        (t->t.Δv).(transfers), 
        xlabel = "Julian Days", 
        ylabel = "Δv",
        marker = 2,
        linealpha = 0,
        xlims = (soonest_launch, soonest_launch + launch_lookahead),
        ylims = (0, maximum(t->t.Δv, transfers)),
        legend = false,
        title = "Δv of Possible Orbits by Date"
    ))  
    
    println("Press enter to view each orbit candidate")
    readline()
    
    transfers = best_transfers(transfers)
    tf_ind = display_and_choose_transfers(transfers, orbit_1, orbit_2, body_1, body_2)
    outbound = transfers[tf_ind]
    
    print("Enter the minimum time to spend here in days: ")
    layover = parse(Float64, readline())
    
    soonest_return = outbound.t_a + layover
    latest_return = soonest_return + launch_lookahead
    
    transfers = find_transfers(
        soonest_return, 
        latest_return, 
        orbit_2, 
        orbit_1, 
        body_2, 
        body_1,
    )
    
    # plot energies
    display(plot(
        (t->t.t_l).(transfers), 
        (t->t.Δv).(transfers), 
        xlabel = "Julian Days", 
        ylabel = "Δv",
        marker = 2,
        linealpha = 0,
        xlims = (soonest_return, latest_return),
        ylims = (0, maximum(t->t.Δv, transfers)),
        legend = false,
        title = "Δv of Possible Orbits by Date"
    ))
    
    println("Press enter to view each orbit candidate")
    readline()
    
    transfers = best_transfers(transfers)
    tf_ind = display_and_choose_transfers(transfers, orbit_2, orbit_1, body_2, body_1)
    inbound = transfers[tf_ind]

    println()
    println("Launch date: ", jd_to_date(outbound.t_l))
    println("Duration along outbound transfer (days): ", outbound.t_a - outbound.t_l)
    println("Time to wait on mars (days): ", inbound.t_l - outbound.t_a)
    println("Duration along inbound transfer (days): ", inbound.t_a - inbound.t_l)
    println("Total journey time (days): ", inbound.t_a - outbound.t_l, ", return date: ", jd_to_date(inbound.t_a))
    println()
    println("Leaving   Δv from GEO to TRANSFER (m/s): ", outbound.leave_Δv)
    println("Arriving  Δv from TRANSFER to GEO (m/s): ", outbound.arrive_Δv)
    println("Returning Δv from GEO to TRANSFER (m/s): ", inbound.leave_Δv)
    println("Final     Δv from TRANSFER to GEO (m/s): ", inbound.arrive_Δv)
    println("Total Δv (m/s): ", outbound.Δv + inbound.Δv)

end

main()
