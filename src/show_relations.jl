using Statistics, Plots, LinearAlgebra

"""
placeholder for real function. only works for berlin, germany, rome, italy, madrid, spain, paris, france
returns 300d embedding for a string
function get_word_embedding(word::String)::Vector
        
        berlin_str ="-0.10052 -0.017 -0.21077 -0.50765 -0.13132 0.21765 -0.26609 0.22856 -0.19865 0.33651 -0.11927 0.12591 0.47005 0.3127 -0.23583 -0.23453 -0.014921 0.07109 0.434 0.57627 0.09705 0.27706 -0.35411 0.19945 0.04167 -0.13174 -0.049887 -0.088114 0.23986 0.19416 -0.34101 0.18284 -0.038464 0.34261 0.20358 0.064062 -0.050313 -0.34356 0.17962 -0.096653 0.23099 -0.11119 -0.14206 0.017161 -0.10122 0.03682 0.0799 -0.031689 0.40153 -0.023915 0.31156 0.021064 -0.29796 -0.11779 0.035163 -0.37074 -0.0058891 0.3212 -0.090729 0.082942 0.28965 -0.10532 0.36011 -0.13221 -0.37254 0.072096 0.19259 0.202 -0.59373 -0.28543 0.27862 -0.31069 -0.27128 0.15297 -0.33777 0.11989 0.40642 0.24152 -0.13345 -0.46217 0.11127 0.47736 0.31819 -0.1213 -0.19735 -0.19502 0.13229 -0.39477 -0.083519 0.069615 0.16632 -0.21403 0.087144 -0.15085 -0.29616 0.25719 -0.46212 0.057489 -0.15393 -0.053855 -0.097108 0.29466 0.10151 -0.0075112 -0.054118 0.077753 -0.21713 0.15798 0.24575 -0.089792 -0.2821 -0.50229 0.39801 -0.31282 0.16383 -0.19914 0.24878 0.18505 -0.10793 0.089811 0.16746 0.14221 -0.02886 0.27972 -0.15821 0.071109 -0.04185 0.32773 0.48422 0.10852 0.0059997 0.10709 -0.052773 -0.53095 -0.13066 -0.032268 -0.21193 -0.14371 -0.49284 0.46708 -0.273 -0.36204 0.062994 -0.20143 0.0027814 0.19583 0.015716 0.17015 -0.0093628 0.16376 -0.073737 -0.3116 -0.18126 -0.35548 -0.28484 -0.12904 -0.062218 -0.2456 -0.52714 0.42998 0.44444 0.62813 0.033329 0.18109 -0.13282 -0.16096 -0.26294 0.31051 0.34078 -0.16779 -0.64985 -0.26081 -0.099533 0.042371 0.10895 0.059171 -0.21959 -0.4401 0.057538 -0.097591 -0.30488 -0.2746 -0.32697 0.040428 0.1185 0.35537 -0.11794 0.044468 -0.17292 0.028476 0.063188 -0.52779 0.061395 0.14078 -0.33479 -0.034258 -0.36712 0.082368 -0.28874 -0.38247 0.86799 0.19181 -0.04134 0.035703 0.32484 -0.51533 -0.58314 0.012628 0.40387 0.055428 0.35311 -0.15689 0.00097598 0.0538 0.19946 -0.37907 -0.12751 -0.51687 -0.32145 -0.20275 -0.096348 0.01612 -0.16761 -0.062865 -0.19366 -0.066753 0.058816 -0.26176 0.43434 -0.24779 0.5044 0.24896 0.26848 0.086134 0.091979 -0.047441 0.37085 0.13166 -0.062037 0.42314 -0.0026331 -0.18796 0.35673 -0.022603 -0.031035 -0.088854 -0.1549 -0.086829 -0.40554 -0.41943 0.22638 0.051392 -0.058888 -0.1738 -0.037833 -0.17967 -0.29044 -0.16025 -0.31979 0.23111 -0.28515 0.21077 0.11575 0.26087 -0.14531 0.048127 -0.0044873 -0.076777 -0.23207 -0.19785 -0.20284 0.058001 -0.076868 -0.15327 -0.064433 -0.14932 -0.02172 0.1706 0.42813 -0.048095 -0.25063 -0.49081 -0.32223 0.25426 -0.22155 0.30175 -0.43747 0.11183 -0.38952 0.51956 -0.20163 -0.082513 0.34158 0.067009 -0.077654 -0.093298 -0.28296 0.045193 -0.068595 0.27074 "
        germany_str = "-0.20947 0.092044 -0.081754 0.057696 -0.33654 -0.047499 -0.22546 -0.26234 -0.11614 0.094206 -0.26419 0.079003 0.21629 0.10931 -0.071479 -0.42733 0.18731 0.51242 0.48695 0.14688 0.20093 0.23443 -0.27167 0.13979 -0.14047 -0.098305 0.0037765 -0.22529 0.18401 0.15886 -0.23973 0.3565 -0.63342 0.22442 -0.07232 -0.27882 -0.091204 -0.23501 0.55748 0.12161 -0.34588 0.089675 -0.36932 0.11233 0.00091398 -0.21387 -0.047945 0.084726 0.19042 -0.025982 0.016842 -0.2146 -0.21773 0.056044 -0.26952 -0.16876 -0.078455 0.24982 -0.31037 0.35461 0.018853 -0.064161 0.20368 -0.36327 -0.18218 -0.10502 -0.12311 0.29273 -0.33193 -0.42501 0.60748 -0.22663 0.10913 0.2697 -0.17387 0.30944 0.52453 0.3085 -0.11993 -0.59885 -0.29101 0.084161 0.26911 -0.095736 -0.059152 -0.13212 -0.083931 -0.072829 0.12647 0.069365 -0.028389 -0.087055 0.10566 0.071886 -0.10843 0.089603 -0.35359 0.012826 -0.22778 -0.039075 0.045487 0.28281 0.29347 -0.090244 -0.18877 -0.087399 -0.23241 -0.067747 0.22759 0.018625 -0.015829 -0.29179 -0.0028244 -0.13975 0.0043034 -0.10286 0.042019 0.0084601 0.32264 0.12718 0.3163 0.25659 -0.19799 0.10853 -0.059033 -0.1509 0.3031 0.16422 0.17654 0.069011 0.024901 0.22933 -0.07795 -0.68017 -0.42027 -0.48179 0.058036 0.11521 0.2299 0.51216 -0.056608 -0.20997 0.14069 0.01958 0.13874 0.53686 0.11833 0.22337 0.14458 0.042719 0.43273 -0.23492 -0.0036791 -0.20618 -0.64581 -0.13513 -0.068283 -0.14001 -0.14334 0.31941 0.21378 0.39915 -0.11209 -0.14097 -0.011734 -0.29899 0.017981 0.4423 0.35001 -0.21241 -0.29527 -0.19854 0.031829 0.35818 -0.025591 -0.17329 0.11605 -0.0073654 -0.0097356 0.22354 -0.17496 -0.74001 -0.23003 0.14444 0.066226 -0.027027 0.025917 -0.12035 -0.063993 0.091126 -0.20611 -0.41944 0.072929 -0.27596 -0.166 -0.023301 0.21495 0.0038261 -0.23519 -0.097132 0.2803 0.044735 0.35735 0.11317 0.32556 -0.20694 -0.36468 -0.48284 0.027713 0.47624 0.56669 0.0057605 -0.17769 0.32429 0.13413 -0.32043 0.14097 -0.40471 -0.12283 -0.14801 -0.17258 0.025419 -0.14673 -0.028935 0.1345 -0.045394 -0.014672 -0.38527 0.19493 -0.4365 0.077317 0.0058776 0.0033151 0.16085 0.12966 0.12729 0.085151 0.17284 -0.043316 0.24693 0.11025 0.22837 0.50621 -0.17364 -0.00048663 -0.090296 -0.18439 0.28459 -0.098643 -0.0036996 0.12965 0.19485 0.098409 -0.27317 0.053153 -0.08996 -0.23039 0.21481 0.15163 0.47065 -0.14852 0.13339 0.093155 -0.27074 -0.20302 -0.12694 0.092938 0.086456 -0.063701 -0.38603 -0.168 0.063726 -0.078884 -0.036706 -0.056929 -0.28633 0.047709 0.4422 0.39222 0.047903 -0.049811 -0.26767 -0.21101 0.3802 -0.024064 0.29455 -0.19669 -0.4264 -0.5694 0.27029 -0.10361 -0.05606 0.071104 0.4373 0.1386 -0.11365 -0.25436 0.04491 -0.076182 0.27178 "    
        rome_str = "0.19442 -0.24911 -0.26938 -0.11867 -0.4707 0.23359 0.17574 0.22614 -0.29807 0.32761 0.3001 -0.1306 0.12557 0.014626 0.10806 -0.35917 -0.011112 0.2606 0.31528 0.23242 -0.2446 0.63753 -0.59869 -0.049033 -0.052985 0.050645 -0.12604 -0.23311 0.21704 -0.10374 -0.17278 0.23054 -0.37512 0.1935 -0.14277 -0.34685 0.019913 -0.5821 0.11574 -0.2463 0.25631 0.020622 0.028892 -0.41613 -0.0069881 -0.36871 0.020451 -0.23738 -0.19602 -0.12815 0.23969 0.0959 -0.13907 -0.0017528 -0.33233 -0.10548 0.13865 -0.064641 -0.36066 0.022961 -0.12867 0.030065 0.1564 -0.12858 -0.050949 -0.24336 -0.031343 0.37511 -0.51228 0.028456 -0.052045 -0.092043 -0.1891 0.06689 -0.067031 0.28669 0.015234 0.46846 -0.19469 -0.39434 -0.41918 0.48228 0.35542 0.089466 -0.51841 -0.083713 0.18524 0.39962 0.16782 -0.16565 -0.092368 -0.70872 0.31903 -0.09511 -0.086334 -0.02276 0.0016352 -0.088429 -0.15585 -0.3031 0.17802 -0.18222 0.28733 0.25613 0.24381 -0.0073827 -0.10379 -0.13544 -0.25906 0.24412 -0.054848 0.01161 0.35282 -0.021756 0.21315 -0.091637 -0.29633 0.35258 0.028917 -0.10351 -0.09505 0.26396 0.42418 -0.091977 -0.20195 0.020757 0.21014 0.29116 0.2474 -0.14548 0.10427 -0.22951 -7.718e-05 -0.2236 -0.12046 0.23452 -0.32007 -0.10091 -0.11109 0.45229 -0.087655 -0.093293 -0.061814 0.11572 0.065992 0.35531 -0.23257 -0.047696 -0.46844 0.49242 0.021462 -0.25731 0.47863 0.10194 0.11719 -0.05189 0.14112 -0.051239 0.12777 0.016591 -0.083278 0.32137 -0.15823 -0.067069 0.11614 -0.064108 -0.047956 0.42092 0.5722 0.011578 -0.41002 -0.053386 -0.16192 0.0062045 0.21225 0.32021 -0.14184 0.057743 0.14191 -0.1885 0.1801 -0.44168 0.09277 -0.24297 -0.16951 -0.031243 -0.089984 -0.051146 -0.018811 -0.28807 -0.067654 -0.91054 0.12575 0.18972 -0.4988 -0.24846 -0.12581 -0.10834 -0.091345 0.27376 0.33733 0.30474 -0.078503 -0.1351 -0.099406 -0.55764 -0.13636 -0.35033 0.32859 -0.11867 0.555 -0.24336 0.059625 0.20565 -0.046047 0.019945 -0.36596 -0.40578 -0.17104 -0.30352 0.072301 -0.26376 -0.38431 -0.090321 0.19003 0.20512 -0.14711 -0.12388 -0.035305 -0.45091 0.43883 0.18136 -0.14554 0.22931 0.22335 -0.04962 0.23721 0.20929 -0.41505 -0.0465 0.049447 -0.54202 -0.13692 0.19473 -0.33752 0.012149 -0.22891 0.07515 0.043001 -0.08942 0.24958 0.47738 0.020088 -0.13259 -0.33046 0.051844 -0.25754 -0.14328 0.22711 0.20196 -0.044893 -0.2187 0.65481 0.14473 -0.12683 -0.016605 0.044344 0.12682 -0.060103 0.13696 -0.085027 -0.1423 -0.37151 -0.39836 -0.096992 -0.1082 0.091131 -0.026229 -0.07031 0.14402 -0.15015 -0.12427 0.10876 0.21503 0.22957 -0.33118 -0.26574 0.26347 0.2745 0.2408 -0.19204 -0.14152 0.1384 0.13752 0.34184 -0.12551 -0.083238 0.20294 0.1669 0.19357 "
        italy_str =  "-0.039592 -0.35959 0.022843 0.24993 -0.33178 0.14435 0.14588 0.0033198 -0.24036 -0.032971 -0.011734 -0.019334 0.22392 0.13854 0.0077642 -0.33037 -0.059784 0.30215 0.20526 -0.11616 0.082219 0.22358 -0.41715 0.11262 -0.39953 -0.31993 0.0056773 -0.43086 0.044288 0.010486 -0.17297 -0.020993 -0.45451 0.035766 0.062142 -0.47654 -0.13884 -0.4833 0.33237 -0.29883 -0.22142 -0.24952 -0.29975 -0.15556 0.044327 -0.29539 0.054063 -0.31341 -0.33047 0.073845 0.039041 -0.36078 -0.15629 0.024584 -0.42783 -0.028739 0.21701 0.12278 -0.36665 0.27444 -0.10853 0.027738 0.21439 -0.1352 -0.019029 -0.11174 -0.23134 0.34055 -0.47386 -0.018413 0.30913 -0.12039 0.19113 0.27074 -0.053042 0.48074 0.31308 0.40945 -0.048661 -0.18328 -0.35154 0.47528 0.33269 -0.16559 -0.53693 -0.10606 -0.016352 0.48451 0.057257 -0.17134 -0.042914 -0.7263 0.15142 -0.054325 0.30031 -0.2132 -0.076102 0.20701 -0.20817 -0.17184 0.0037446 -0.1308 0.22343 0.064955 -0.024532 -0.27672 -0.22727 -0.061838 0.068346 0.060623 0.079675 -0.048984 0.06157 0.24886 0.13177 0.29966 -0.065853 0.088467 0.27619 0.022752 0.017582 0.046502 0.17293 -0.095077 -0.020484 0.020939 0.46073 0.1333 -0.046523 -0.040394 0.093419 0.11608 -0.15205 -0.4733 -0.31495 -0.30817 -0.16033 0.24186 0.22518 0.36869 0.068723 -0.071967 0.052871 0.21219 0.086971 0.74024 0.072041 0.13036 -0.30993 0.33444 0.40511 -0.32768 0.21127 -0.0057938 -0.34258 -0.084268 0.19473 -0.27633 0.13489 0.14602 -0.20712 0.11916 0.049906 -0.11527 0.44864 -0.096141 -0.067828 0.39376 0.40379 0.078891 -0.13543 0.027367 0.22532 0.058189 0.092435 0.031676 0.1145 -0.00042753 0.25419 0.29561 0.022016 -0.70146 0.015399 0.17836 0.11286 -0.32756 -0.38725 0.04632 0.10488 -0.23261 -0.33262 -0.41117 0.26374 0.080949 -0.42783 -0.16811 0.16201 -0.11581 -0.27895 0.033704 0.10976 0.13983 0.35457 -0.023984 0.16469 -0.3365 -0.03578 -0.38865 0.23345 0.13951 0.46366 -0.33151 -0.079028 0.20477 0.34418 0.03665 -0.069701 -0.33336 -0.37761 -0.1685 0.37097 -0.27558 -0.056326 0.22921 0.25228 -0.19721 -0.35481 -0.32935 -0.0032477 -0.52265 0.12525 0.20196 -0.35402 0.12937 0.056696 0.064719 -0.10358 0.23164 -0.25607 0.0042785 0.51821 -0.091204 0.14396 0.095985 -0.17241 -0.058219 -0.15915 0.31522 -0.093068 0.040268 0.18486 0.16337 0.0087755 -0.18849 -0.16834 -0.083769 -0.10698 0.024818 0.40294 0.52261 0.082893 -0.13758 0.52907 -0.38064 -0.29612 0.11962 0.14747 0.060694 -0.072687 0.14194 -0.25241 -0.27264 -0.091267 -0.10209 -0.017087 -0.34331 0.15202 0.42045 0.18712 0.066572 -0.082668 0.027632 0.25968 0.42341 0.097947 -0.17528 -0.17605 -0.26879 0.052509 0.23228 -0.30326 -0.23892 0.24874 0.098513 0.41338 -0.13436 -0.10317 0.21119 -0.084965 0.30524"
        madrid_str = "0.24014 0.21532 -0.19113 -0.42414 0.29127 0.25442 -0.087802 -0.057619 -0.077056 0.1133 0.4678 -0.35976 0.30382 0.32432 -0.33916 -0.38502 -0.26202 0.15677 0.41634 0.55077 -0.23705 0.25225 -0.34566 -0.089176 -0.08976 -0.1526 -0.098455 -0.18386 0.20796 0.0069461 0.031636 -0.096829 0.059677 0.3381 -0.16724 0.049215 0.12052 -0.46889 0.31167 0.13898 0.47506 0.097157 -0.173 -0.11655 0.11015 -0.15571 0.093819 -0.42753 0.14196 0.031749 0.47398 -0.221 -0.21443 0.29 -0.33725 -0.11439 0.22274 0.40122 -0.46598 0.39561 -0.23738 -0.29058 0.27213 0.062892 0.20376 0.021004 0.066835 0.040411 -0.29868 -0.19917 -0.057432 0.021166 0.26832 0.020737 -0.19068 0.015326 0.60755 0.33609 -0.0058094 -0.24219 -0.10528 0.80025 0.18352 -0.068349 0.091789 -0.13461 0.17749 0.18425 0.19852 0.013209 -0.21701 -0.65167 0.27332 0.068794 -0.10805 0.21107 -0.25535 0.37754 -0.45484 0.0038515 0.097143 -0.20961 0.42688 -0.25942 0.30045 -0.10238 -0.15347 0.084744 0.11935 -0.38674 -0.036748 -0.012158 0.57321 0.20445 0.095709 0.15113 0.28024 0.30723 0.29906 -0.10246 -0.035658 -0.31786 0.3533 -0.21242 -0.15208 -0.15862 0.11801 0.35699 0.40041 0.050323 -0.19392 -0.34225 0.30593 0.21775 -0.30632 0.23444 -0.15931 0.13763 -0.086819 0.73327 -0.44047 -0.24973 -0.25983 0.062714 0.05321 0.22326 0.25122 -0.083187 0.10529 0.25977 -0.23708 -0.33335 -0.086898 0.020603 -0.11852 -0.38918 0.1075 -0.090757 -0.57015 0.45442 0.056583 0.05644 -0.21863 0.36319 -0.38503 0.28023 -0.10121 0.39553 0.49789 0.098399 0.18961 0.13299 -0.20498 0.054836 0.19301 0.22668 -0.10993 0.13377 -0.35007 -0.34468 -0.2931 0.06824 -0.31415 0.054017 0.35347 0.41715 -0.072924 0.19104 -0.068924 0.106 -0.12049 -0.15893 0.45507 0.24366 -0.46153 -0.32322 -0.041621 0.02871 -0.31715 -0.24824 0.60459 0.1902 0.23543 -0.2804 0.52815 -0.5988 -0.37444 -0.17878 0.20187 -0.029911 0.77196 0.33389 0.46983 0.06473 0.012509 -0.19926 -0.24976 -0.13787 -0.17445 -0.062846 0.18707 -0.14076 -0.18794 0.47203 0.33622 0.21685 -0.087957 0.21226 0.28893 -0.069434 0.33794 0.21076 -0.049943 0.035451 -0.18001 -0.090074 0.077309 0.087981 0.17179 0.017449 0.37651 -0.13459 0.061128 -0.048039 -0.5468 -0.3168 0.11544 0.043722 -0.25811 -0.36557 0.12137 0.017438 0.04466 -0.34214 0.06816 -0.20483 -0.47985 -0.08738 0.57167 0.30011 0.18276 0.13694 0.60042 0.22243 -0.52841 -0.17418 -0.041747 0.091712 -0.14678 -0.090161 0.096208 0.032737 0.034501 -0.27048 -0.73684 -0.57526 0.16871 0.44967 0.099882 0.095963 -0.063685 0.073731 -0.46428 0.28131 -0.13469 -0.29997 -0.069007 0.12216 0.24953 0.23667 -0.16915 -0.36557 0.21899 0.085461 0.29099 -0.27717 -0.1059 -0.077028 0.50645 -0.32713 "
        spain_str = "-0.19976 0.22547 -0.24129 -0.20261 -0.092315 0.21181 -0.03758 -0.16241 0.061027 0.26185 0.24168 -0.28113 0.22518 0.24378 -0.32316 -0.43269 -0.16673 0.33961 0.35906 0.28415 -0.11034 0.08933 -0.22962 -0.075711 -0.52757 -0.19388 -0.12459 -0.47735 0.44837 0.18808 0.072942 0.083304 -0.37965 0.21462 -0.19645 0.13921 0.15707 -0.3962 0.4457 0.07899 0.080613 -0.34499 -0.22559 -0.2 0.19259 -0.18968 0.025639 -0.36388 0.12644 0.14987 0.27473 -0.10421 -0.057436 0.13511 -0.31228 -0.21216 0.14996 0.39464 -0.52609 0.66834 -0.24232 -0.0011246 0.31777 -0.090309 0.17764 0.049792 -0.35502 -0.14292 -0.42587 -0.18776 0.32609 -0.084013 0.428 0.097894 -0.24377 0.06372 0.43776 0.60419 -0.065335 -0.19791 -0.36433 0.31746 0.056866 -0.072734 -0.012558 -0.24841 -0.11512 0.4455 0.22836 0.024194 -0.27143 -0.51526 0.25489 -0.10946 0.36669 0.068119 -0.23709 0.45348 -0.082474 -0.18297 0.32126 -0.211 0.1185 -0.22762 0.19337 -0.092391 -0.11746 0.19596 0.17413 -0.25264 0.16505 0.015019 0.010724 0.067675 -0.011006 0.42067 0.041565 0.096693 0.53921 -0.01346 -0.12541 -0.086049 0.29457 -0.25772 -0.16372 -0.079197 0.25082 0.18629 0.068817 0.12487 0.17456 0.034661 0.042186 0.045933 -0.28287 -0.10184 0.0021812 0.29953 0.2048 0.75641 -0.017451 0.08689 0.038819 0.0040263 -0.078779 0.48627 0.054624 -0.18356 -0.0062347 0.28353 0.071469 -0.46386 -0.10236 -0.024363 -0.36115 -0.097578 -0.0060282 0.013384 -0.10932 0.29235 -0.25394 -0.056667 0.054601 0.15148 -0.24177 0.045195 -0.074451 0.15539 0.18911 0.031388 0.15458 -0.10154 -0.035173 0.12958 -0.018958 -0.1373 0.10867 0.31909 -0.19286 0.2863 0.021779 -0.67512 -0.14535 0.1694 0.16121 -0.065057 0.057494 0.10712 -0.17845 -0.092673 -0.27227 -0.21626 0.40303 0.14351 -0.10411 -0.098407 -0.043946 -0.018707 -0.45662 0.14715 0.51194 0.054677 0.36498 0.087258 0.34116 -0.42932 -0.12919 -0.21341 0.092247 0.39027 0.58046 0.16212 0.11331 0.037002 0.13579 -0.20142 0.30502 -0.12405 -0.20049 0.028271 0.086086 -0.2724 -0.19092 0.30974 0.44705 0.15387 -0.078865 -0.087396 -0.093979 -0.28274 0.14824 0.090374 -0.19605 0.16235 -0.22351 -0.019319 -0.2047 0.10773 0.061492 -0.16974 0.34372 -0.061758 0.21727 -0.10054 -0.22827 -0.31297 0.0071546 0.39848 -0.021631 -0.022985 0.20805 0.04524 0.16727 -0.18066 -0.32838 -0.2014 -0.24861 0.14177 0.57984 0.3862 0.24323 -0.1894 0.64011 -0.65064 -0.60916 -0.055023 -0.016392 0.35177 -0.20574 0.049733 0.044235 -0.12121 0.043476 -0.32609 -0.52466 -0.509 0.11767 0.38256 0.34081 0.04572 -0.16275 0.016248 -0.094992 0.17954 -0.081788 -0.020457 0.077413 -0.34048 -0.11394 -0.10665 -0.33909 -0.38573 -0.086062 0.098092 0.48132 -0.04984 0.26661 0.045438 0.069751 0.09833 "
        paris_str = "0.1286 -0.17759 -0.27576 -0.47671 -0.12229 -0.03844 0.095767 -0.14726 -0.10583 0.35433 0.39459 -0.21991 0.17539 -0.007593 -0.20857 -0.025608 -0.069811 0.24765 0.26921 0.45075 0.0071364 0.46008 -0.53388 0.14545 -0.114 -0.070477 -0.1308 -0.15418 0.35926 0.31601 -0.34452 0.2045 0.15459 0.35185 -0.11769 0.053657 -0.10284 -0.44093 -0.13058 0.02343 -0.039494 -0.071015 -0.13643 -0.27216 -0.12194 -0.26531 -0.16812 -0.33127 -0.063204 0.25064 0.45215 0.060244 -0.094094 0.062653 -0.29754 -0.27591 0.34969 -0.1869 0.20278 -0.074274 0.16125 -0.015935 0.031196 -0.15776 -0.27292 0.06907 0.17524 0.12008 -0.27017 0.071333 0.019069 0.081177 -0.26407 0.097202 -0.021865 0.081035 0.36303 0.17389 0.23979 -0.087882 -0.19522 0.13474 0.128 -0.054796 -0.36088 -0.23172 -0.014965 -0.10519 0.074197 0.20344 -0.25497 -0.32781 0.1097 -0.043131 0.024719 0.17277 -0.082305 0.094167 -0.33669 -0.020538 -0.22014 -0.060063 0.23313 0.068187 0.13248 -0.074956 -0.13864 -0.25084 0.12599 0.25193 0.25641 -0.002246 0.42571 0.09311 -0.066148 -0.081437 0.15792 0.024168 0.20227 -0.07105 0.074295 0.055801 0.15149 0.16714 -0.11868 0.085014 0.2196 0.47768 0.21648 0.28113 0.072566 -0.047657 -0.076406 -0.27205 -0.07902 0.38736 -0.28183 0.10963 -0.4104 0.33116 -0.34688 -0.016262 0.23779 0.041696 0.01309 0.14072 0.28256 -0.0027894 -0.10705 0.20619 0.41699 -0.56284 -0.065843 -0.20456 -0.045762 -0.2113 -0.3224 -0.23153 -0.0033393 0.32108 0.026557 0.47823 0.095614 0.3341 0.09097 -0.021247 0.049389 0.080207 0.76284 -0.10895 -0.042168 0.08568 -0.10335 -0.32381 0.28989 0.46821 -0.28171 0.1632 -0.13918 -0.24323 0.027113 -0.25755 -0.012403 0.098824 -0.17044 0.40899 -0.15544 0.079638 -0.062305 -0.21229 -0.030271 -0.41519 0.10786 0.32948 -0.87099 -0.052573 0.046852 0.25266 -0.076092 -0.36537 0.22671 0.36047 -0.049665 -0.27929 0.22665 -0.76824 -0.61379 -0.21269 0.37252 0.012927 0.27305 0.098732 0.17561 0.11531 -0.0804 -0.070489 -0.37199 -0.20228 -0.24743 0.021289 0.27743 0.29915 -0.20615 0.074367 -0.15922 0.19015 -0.090382 -0.071185 0.33812 -0.2276 0.3234 0.12948 -0.023147 0.11347 -0.13666 -0.18222 0.23963 -0.015457 -0.30481 0.34548 -0.29666 -0.36829 0.44238 0.19866 -0.11465 0.19832 0.38023 0.082262 -0.19678 -0.10015 0.092372 -0.18626 -0.20012 -0.1208 -0.10878 -0.10765 -0.27768 -0.076544 -0.044823 0.24123 0.11465 -0.0087452 0.33234 0.096864 -0.17126 0.040056 -0.27986 0.14021 -0.10694 0.42879 0.17129 -0.01582 -0.075588 -0.33389 -0.35291 -0.23693 -0.27407 0.15886 0.08963 0.27708 -0.47897 -0.16879 -0.15245 0.36916 -0.28942 0.405 -0.20926 -0.030516 0.12824 0.39265 -0.005261 -0.23109 0.095161 0.060371 0.052252 -0.091958 -0.27226 0.13525 0.17608 -0.045033 "
        france_str = "-0.17135 -0.026687 -0.15725 -0.0040515 -0.30934 -0.049739 0.13636 -0.24766 -0.018268 0.15679 0.14284 -0.11148 0.163 -0.20161 -0.10348 -0.08172 -0.021883 0.42307 0.49761 0.18683 0.099254 0.0043156 -0.41715 0.2109 -0.32355 -0.075391 -0.039346 -0.54813 0.32632 0.37267 -0.4347 0.21834 -0.39394 -0.11458 -0.0025308 -0.1699 -0.13909 -0.32504 0.043649 0.24111 -0.31168 -0.36608 -0.16974 -0.051673 0.010993 -0.24851 -0.1389 -0.22241 -0.088253 0.29723 0.24621 -0.049026 -0.15807 0.083506 -0.16849 -0.12105 0.38397 -0.14702 -0.15301 0.54084 -0.20134 0.13382 -0.005902 -0.24942 0.05416 -0.067202 -0.29265 0.20881 -0.59006 -0.32917 0.25125 0.045963 0.19667 0.23794 -0.0039842 0.17179 0.61426 0.28097 0.14259 -0.18375 -0.26579 0.064082 -0.0064737 -0.071906 -0.34928 -0.33582 -0.1226 0.33202 0.21369 0.11011 -0.45343 -0.21263 0.14204 -0.093156 0.34603 0.042735 -0.086029 0.044733 -0.4249 -0.041276 -0.28119 -0.083311 0.04223 -0.26302 0.078205 -0.40102 -0.21717 -0.12413 0.1001 -0.042123 0.23175 -0.084368 0.075728 0.13247 -0.1076 0.12625 0.26939 -0.081202 0.52163 0.087973 0.067796 -0.058463 0.1227 0.16449 0.058358 -0.15886 0.45535 0.34117 -0.039575 0.28314 0.042863 0.04967 -0.091396 -0.54902 -0.14935 -0.027113 -0.19181 0.23988 0.39774 0.47523 -0.09889 0.0014672 0.29002 0.097363 0.056992 0.43284 0.24067 0.099709 0.036022 0.13618 0.46329 -0.53018 -0.038699 0.013618 -0.30446 0.15022 -0.18819 -0.27739 0.13361 0.23134 -0.10133 0.17766 0.39425 0.069185 0.15706 0.022259 0.23905 0.17781 0.43408 -0.25532 0.18029 0.1752 0.016844 -0.083187 0.067043 0.20729 -0.14495 0.12877 -0.079838 0.048367 0.10122 -0.57097 -0.11782 0.3895 0.015691 -0.015202 -0.088832 -0.043099 -0.056443 -0.23359 -0.21645 -0.47341 0.050991 0.11449 -0.58241 0.12256 0.39144 -0.17025 -0.36128 -0.17289 0.084306 0.065699 0.14807 -0.046399 0.42764 -0.60642 -0.28379 -0.23006 0.072331 0.32837 0.23139 0.006793 -0.21091 0.073637 -0.15385 0.095075 0.12465 -0.25233 -0.045604 0.098714 0.29409 -0.0026719 -0.19072 0.045793 0.14954 0.03306 -0.30391 -0.15362 0.055417 -0.55288 -0.15294 0.0038574 -0.26377 0.10363 -0.069797 0.048029 -0.031764 0.22006 -0.36167 0.16044 0.020909 -0.095996 0.44367 0.23154 0.024462 0.33358 0.30206 0.4571 -0.096147 0.063791 0.17035 -0.14947 0.10285 0.049313 -0.12738 -0.15748 -0.072423 0.11883 0.17248 0.48882 -0.0090107 -0.10323 0.38906 -0.2334 -0.3793 -0.13098 -0.11088 0.25865 -0.17604 0.20591 -0.11252 -0.23044 -0.0044065 -0.40153 -0.06399 -0.47027 -0.15099 0.23548 0.31269 0.22888 -0.39532 -0.074453 -0.016224 0.15296 -0.18036 0.44396 0.028257 -0.57095 -0.25477 0.19632 -0.045836 -0.34219 -0.20158 0.23946 -0.003533 0.073969 -0.039123 0.093385 -0.071573 0.028744 "
        testable_str = "-0.18306 -0.050346 -0.3497 0.55406 0.073992 0.15294 0.16146 -0.53301 -0.43052 0.26416 0.1285 0.35976 -0.33245 -0.065106 0.22569 -0.01164 0.14293 0.31131 0.17955 0.21715 0.16726 -0.28377 0.0828 -0.50235 -0.39922 0.0069105 0.05105 0.01857 0.098398 0.24151 -0.12344 0.34723 -0.25558 0.4878 0.094184 0.14207 0.17574 0.063384 -0.32223 0.25758 0.0028367 0.1169 -0.0039779 0.050848 -0.074859 0.37523 0.31568 -0.25066 0.13314 -0.14776 0.13514 -0.071088 -0.28477 -0.47105 -0.15173 -0.16768 -0.23262 0.33856 0.40293 0.44876 -0.12355 -0.15073 0.034162 -0.035373 -0.31423 -0.69999 0.02605 0.05276 0.12243 0.21469 0.27197 0.25742 0.28577 -0.10682 0.15879 0.038959 0.027427 -0.057872 0.069669 -0.0051221 0.34849 0.41272 0.087703 -0.1787 -0.23578 0.07269 0.46063 0.093011 -0.2102 -0.14714 -0.084243 0.14274 0.52417 -0.47492 -0.39695 -0.2843 0.50576 0.26736 -0.098888 0.25529 0.046507 -0.44295 0.066284 -0.22984 0.016442 0.43832 -0.1954 -0.34192 -0.058725 -0.35088 -0.050514 0.062977 -0.42805 0.17707 0.059244 -0.33745 0.14404 0.029227 0.022174 -0.027444 -0.10768 -0.16445 -0.25651 0.55196 -0.064191 -0.089096 0.17929 0.329 -0.053746 -0.017813 -0.024495 -0.26561 -0.34252 -0.090646 0.32505 -0.1433 -0.39521 -0.0022949 0.17311 0.11645 0.088354 -0.24825 0.18287 -0.31676 -0.046251 0.16674 -0.072015 -0.024578 -0.32972 0.10615 0.37499 -0.42406 0.077808 -0.13984 -0.363 0.5136 0.30651 0.023571 0.020956 0.37414 -0.21213 0.32295 -0.16227 -0.075353 0.077426 0.015754 0.087491 -0.41114 0.38083 0.36014 -0.22738 -0.5286 -0.068753 0.27643 -0.068848 -0.045508 -0.48135 0.021122 -0.25306 -0.028359 -0.41776 0.031666 0.61195 0.22745 -0.15424 0.10296 0.030163 0.037252 0.56213 -0.16998 -0.15331 -0.14333 0.19281 0.32069 -0.11916 0.013709 0.089401 -0.022063 -0.34077 -0.046001 0.26448 0.36348 0.21717 0.024835 -0.18256 -0.12928 -0.092963 -0.33312 0.35298 0.066247 0.36576 -0.1152 0.47075 -0.17008 0.064616 -0.17428 0.024732 -0.48209 -0.10218 0.15145 0.046176 -0.11286 -0.2179 -0.092007 -0.10879 0.06899 -0.30444 -0.22212 -0.44394 -0.18965 -0.22206 -0.33271 -0.14582 0.010854 0.076592 0.058845 0.1946 -0.027014 -0.16288 -0.026373 0.025595 -0.18479 0.35536 0.062792 -0.277 0.15084 0.26592 0.10832 0.12782 0.094561 -0.53985 0.11716 -0.35717 0.36404 0.34594 0.15459 0.42673 0.079186 -0.012179 -0.12403 -0.50589 -0.023049 -0.1387 0.49063 -0.29005 -0.015619 -0.21676 0.1832 -0.28775 0.23516 -0.49671 -0.0075416 0.47158 0.03153 -0.39145 -0.070164 -0.078153 0.067227 -0.22808 0.090336 0.042741 0.23545 -0.062233 -0.35928 -0.054606 0.071136 0.57157 -0.045337 -0.20367 0.19335 0.054158 0.10867 0.0090608 -0.081911 0.24633 -0.13805 -0.13262 0.11823 0.34092 -0.20919 "
        berlin, germany, rome, italy, madrid, spain, paris, france, testable = [parse.(Float64, split(str)) for str in [berlin_str, germany_str, rome_str, italy_str, madrid_str, spain_str, paris_str, france_str, testable_str]]
        
        if word=="berlin"
                embedding = berlin
        elseif word=="germany"
                embedding = germany
        elseif word=="paris"
                embedding = paris
        elseif word=="france"
                embedding = france
        elseif word=="spain"
                embedding = spain
        elseif word=="italy"
                embedding = italy
        elseif word=="rome"
                embedding = rome
        elseif word=="madrid"
                embedding = madrid
        elseif word=="testable"
                embedding = testable
        
        else 
                error("no embedding preloaded")
        end
        
        return embedding
end
"""

"""
This function reduces the dimension of a matrix from NxM to Nx"number_of_pc" with a PCA. 
It returns the projected data.
"""
function reduce_to_2d(data::Matrix, number_of_pc=2::Int)::Matrix
        # Center the data
        num = size(data)[1]
        c_data = data .- sum(data, dims = 1) ./ num

        # Compute the covariance matrix
        cov_matrix = cov(c_data)

        # Perform eigen decomposition
        eigen_vals, eigen_vecs = eigen(cov_matrix)

        # Sort eigenvalues (and corresponding eigenvectors) in descending order
        idx = sortperm(eigen_vals, rev=true)
        eigen_vecs = eigen_vecs[:, idx]

        # Select the top 2 principal components
        pca_components = eigen_vecs[:, 1:number_of_pc]

        # Project the data onto the top 2 principal components
        projected_data = pca_components' * c_data'  
        return projected_data
end


"""
This Function creates a plot of the relations of the arguments like this:
arg1==>arg2,
arg3==>arg4,
...
Note: Use an even number of inputs!
"""
function show_relations(words::String...; wv::WordEmbedding, save_path::String="word_relations.png")
    # Check input
    word_count = length(words)
    if Bool(word_count%2)
        throw(error("need words in multiples of 2 but $word_count are given"))
    end
    
    # Check if all words exist in the embedding
    for word in words
        if !haskey(wv.word_indices, word)
            throw(error("Word '$word' not found in embeddings"))
        end
    end
    
    # Get embeddings by looking up each word's index and getting its vector
    embeddings = reduce(vcat, transpose.([wv.embeddings[:, wv.word_indices[word]] for word in words]))
    labels = text.([word for word in words], :bottom)
    
    # Rest of the function remains the same
    # reduce dimension
    projection = reduce_to_2d(embeddings)
    
    # preparation for plotting the arrows, infill with zeros and split x, y
    arrows = [projection[:, 2*i]-projection[:, 2*i-1] for i in 1:Int(word_count/2)]
    arrows_x = [Bool(i%2) ? arrows[Int(i/2+0.5)][1] : 0 for i in 1:length(arrows)*2]
    arrows_y = [Bool(i%2) ? arrows[Int(i/2+0.5)][2] : 0 for i in 1:length(arrows)*2]
        
        p = scatter(projection[1, :], projection[2, :], 
                title="PCA Projection to 2D",
                xlabel="first principal component",
                ylabel="second principal component",
                legend=false, series_annotations = labels)
    
    # plot 2d embeddings
    #gr()
    #scatter(projection[1, :], projection[2, :], 
    #title="PCA Projection to 2D",
    #xlabel="first principal component",
    #ylabel="second principal component",
    #legend=false, series_annotations = labels)
    
    # plot the arrows
        quiver!(p, projection[1, :], projection[2, :], quiver=(arrows_x, arrows_y))

        # Save the plot
        savefig(p, save_path)
end

"""
function show_relations(wv::WordEmbedding, words::String...)
        #check input
        word_count = length(words)
        if Bool(word_count%2)
                throw(error("need words in multiples of 2 but $word_count are given"))
        end

        # Check if all words exist in the embedding
        for word in words
            if !haskey(word_embedding.word_indices, word)
                    throw(error("Word '$word' not found in embeddings"))
                end
        end
        
        #get embeddings and list labels
        embeddings = reduce(vcat,transpose.([get_word_embedding(word) for word in words]))
        labels = text.([word for word in words], :bottom)
        
        # reduce dimension
        projection = reduce_to_2d(embeddings)

        #preperation for plotting the arrows, infill with zeros and split x, y
        arrows = [projection[:, 2*i]-projection[:, 2*i-1] for i in 1:Int(word_count/2)]
        arrows_x = [Bool(i%2) ? arrows[Int(i/2+0.5)][1] : 0 for i in 1:length(arrows)*2]
        arrows_y = [Bool(i%2) ? arrows[Int(i/2+0.5)][2] : 0 for i in 1:length(arrows)*2]

        # plot 2d embeddings
        gr()
        scatter(projection[1, :], projection[2, :], 
        title="PCA Projection to 2D",
        xlabel="first principal component",
        ylabel="second principal component",
        legend=false, series_annotations = labels)

        # plot the arrows
        quiver!(projection[1, :], projection[2, :], quiver=(arrows_x, arrows_y))
end
"""
# show_relations("berlin", "germany", "paris", "testable", "madrid", "spain")

