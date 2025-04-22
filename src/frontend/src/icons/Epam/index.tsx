import React, { forwardRef } from "react";
import EpamSvg from "./EpamSvg";

export const EpamIcon = forwardRef<
    SVGSVGElement,
    React.PropsWithChildren<{ color?: string }>
>((props, ref) => {
    return <EpamSvg ref={ref} {...props} />;
});
