function resp = shortdipole(pax,ang)

resp.H = pax(1)*getShortDipoleHResponse(ang,'X') + ...
         pax(2)*getShortDipoleHResponse(ang,'Y') + ...
         pax(3)*getShortDipoleHResponse(ang,'Z');
resp.H = resp.H*sqrt(3/2);

resp.V = pax(1)*getShortDipoleVResponse(ang,'X') + ...
         pax(2)*getShortDipoleVResponse(ang,'Y') + ...
         pax(3)*getShortDipoleVResponse(ang,'Z');
resp.V = resp.V*sqrt(3/2);

end

function g = getShortDipoleHResponse(ang,axis)
    switch axis
        case 'X'
            g = sind(ang(1,:).');
        case 'Y'
            g = -cosd(ang(1,:).');
        case 'Z'
            g = zeros(size(ang,2),1);
    end
end

function g = getShortDipoleVResponse(ang,axis)
    switch axis
        case 'X'
            g = cosd(ang(1,:).').*sind(ang(2,:).');
        case 'Y'
            g = sind(ang(1,:).').*sind(ang(2,:).');
        case 'Z'
            g = -cosd(ang(2,:).');
    end
end